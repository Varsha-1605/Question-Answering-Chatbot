from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from ai_workflow.cache.semantic import SemanticAnswerStore
from ai_workflow.cache.stats import CacheStats
from ai_workflow.cache.store import CircuitBreaker
from ai_workflow.cache.versions import DocumentVersions


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeIndex:
    def __init__(self, redis):
        self._redis = redis

    def create_index(self, schema, definition=None):
        self._redis.index_creations += 1

    def search(self, query, query_params=None):
        self._redis.searches += 1
        if self._redis.error:
            raise self._redis.error
        return SimpleNamespace(docs=self._redis.docs)


class FakePipeline:
    def __init__(self, redis):
        self._redis = redis

    def hset(self, key, mapping):
        self._redis.hashes[key] = mapping

    def expire(self, key, seconds):
        self._redis.expiry[key] = seconds

    def execute(self):
        if self._redis.error:
            raise self._redis.error


class FakeRedis:
    """Minimal fake of redis-py."""

    def __init__(self):
        self.error = None
        self.docs = []
        self.searches = 0
        self.index_creations = 0
        self.hashes = {}
        self.expiry = {}
        self.values = {}

    def ft(self, name):
        return FakeIndex(self)

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    def set(self, key, value, nx=False, ex=None):
        if self.error:
            raise self.error
        if nx and key in self.values:
            return None
        self.values[key] = value.encode()
        return True

    def get(self, key):
        if self.error:
            raise self.error
        return self.values.get(key)


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def breaker():
    return CircuitBreaker(open_seconds=30, clock=FakeClock())


@pytest.fixture
def store(redis, breaker):
    return SemanticAnswerStore(
        lambda: redis, CacheStats(), breaker, key_prefix="t", threshold=0.9, ttl_seconds=60
    )


def doc(distance, question="earlier question", answer="earlier answer"):
    return SimpleNamespace(question=question, answer=answer, distance=str(distance))


VECTOR = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------- circuit breaker
def test_breaker_opens_on_failure_and_closes_after_the_timer():
    clock = FakeClock()
    breaker = CircuitBreaker(open_seconds=30, clock=clock)
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()
    clock.now += 31
    assert breaker.allow()


def test_breaker_success_closes_it_early():
    breaker = CircuitBreaker(open_seconds=30, clock=FakeClock())
    breaker.record_failure()
    breaker.record_success()
    assert breaker.allow()


# ---------------------------------------------------------------- lookup
def test_similar_enough_question_is_a_hit(store, redis):
    redis.docs = [doc(0.05)]
    hit = store.lookup("scope", VECTOR)
    assert (hit.answer, hit.similarity) == ("earlier answer", 0.95)


def test_question_below_the_threshold_is_a_miss(store, redis):
    redis.docs = [doc(0.3)]
    assert store.lookup("scope", VECTOR) is None


def test_no_stored_questions_is_a_miss(store):
    assert store.lookup("scope", VECTOR) is None


def test_index_is_created_once_per_vector_size(store, redis):
    store.lookup("scope", VECTOR)
    store.lookup("scope", VECTOR)
    assert redis.index_creations == 1


def test_redis_outage_is_a_miss_and_stops_further_calls(store, redis):
    redis.error = RedisConnectionError("down")
    assert store.lookup("scope", VECTOR) is None
    assert redis.searches == 1

    redis.error = None
    redis.docs = [doc(0.0)]
    assert store.lookup("scope", VECTOR) is None  # breaker open: Redis is not even asked
    assert redis.searches == 1


def test_redis_without_vector_search_disables_the_store(store, redis, breaker):
    redis.error = ResponseError("unknown command 'FT.SEARCH'")
    assert store.lookup("scope", VECTOR) is None

    redis.error = None
    redis.docs = [doc(0.0)]
    assert store.lookup("scope", VECTOR) is None
    assert redis.searches == 1
    assert breaker.allow()  # a setup problem, not an outage


# ---------------------------------------------------------------- store
def test_store_saves_hashed_scope_with_a_ttl(store, redis):
    store.store("team-a|v1", VECTOR, "what is rag", "an answer")
    (key, fields), = redis.hashes.items()
    assert fields["question"] == "what is rag" and fields["answer"] == "an answer"
    assert "team-a" not in fields["scope"]
    assert redis.expiry[key] == 60


def test_same_question_in_same_scope_overwrites_instead_of_duplicating(store, redis):
    store.store("scope", VECTOR, "q", "first")
    store.store("scope", VECTOR, "q", "second")
    assert len(redis.hashes) == 1
    assert next(iter(redis.hashes.values()))["answer"] == "second"


def test_same_question_in_another_scope_is_stored_separately(store, redis):
    store.store("scope-a", VECTOR, "q", "a")
    store.store("scope-b", VECTOR, "q", "b")
    assert len(redis.hashes) == 2


def test_store_failure_is_swallowed_and_opens_the_breaker(store, redis, breaker):
    redis.error = RedisConnectionError("down")
    store.store("scope", VECTOR, "q", "a")
    assert not breaker.allow()


# ---------------------------------------------------------------- document versions
@pytest.fixture
def versions(redis, breaker):
    return DocumentVersions(lambda: redis, breaker, key_prefix="t", ttl_seconds=60)


def test_token_is_stable_until_bumped(versions):
    first = versions.current("Idx/ns")
    assert first and versions.current("Idx/ns") == first
    assert versions.bump("Idx/ns")
    assert versions.current("Idx/ns") != first


def test_scopes_have_independent_tokens(versions):
    before = versions.current("Idx/a")
    versions.bump("Idx/b")
    assert versions.current("Idx/a") == before


def test_lost_token_is_replaced_never_defaulted(versions, redis):
    first = versions.current("Idx/ns")
    redis.values.clear()  # evicted by Redis
    assert versions.current("Idx/ns") not in (None, first)


def test_unreadable_version_means_do_not_use_the_cache(versions, redis):
    redis.error = RedisConnectionError("down")
    assert versions.current("Idx/ns") is None


def test_bump_is_still_attempted_when_the_breaker_is_open(versions, breaker, redis):
    breaker.record_failure()
    assert versions.current("Idx/ns") is None
    assert versions.bump("Idx/ns")
    assert redis.values


def test_failed_bump_reports_false(versions, redis):
    redis.error = RedisConnectionError("down")
    assert versions.bump("Idx/ns") is False
