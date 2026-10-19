import hashlib
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import redis
from loguru import logger
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError

from app.cache.stats import CacheStats
from app.cache.store import CircuitBreaker

SEMANTIC_ANSWER_CACHE = "semantic_answer"


@dataclass(frozen=True)
class CachedAnswer:
    question: str
    answer: str
    similarity: float


class SemanticAnswerStore:
    def __init__(
        self,
        client_factory: Callable[[], redis.Redis],
        stats: CacheStats,
        breaker: CircuitBreaker,
        *,
        key_prefix: str,
        threshold: float,
        ttl_seconds: int,
    ) -> None:
        self._client_factory = client_factory
        self._stats = stats
        self._breaker = breaker
        self._key_prefix = key_prefix
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._indexed_dimensions: set[int] = set()
        self._unsupported = False

    def lookup(self, scope: str, vector: Sequence[float]) -> Optional[CachedAnswer]:
        """Return the nearest stored answer if it meets the similarity threshold."""
        if self._unsupported or not self._breaker.allow():
            self._stats.record_bypass(SEMANTIC_ANSWER_CACHE, 1)
            return None

        started = time.perf_counter()
        try:
            self._ensure_index(len(vector))
            query = (
                Query(f"(@scope:{{{_hash(scope)}}})=>[KNN 1 @embedding $vector AS distance]")
                .return_fields("question", "answer", "distance")
                .sort_by("distance")
                .dialect(2)
            )
            result = self._client_factory().ft(self._index_name(len(vector))).search(
                query, query_params={"vector": _to_bytes(vector)}
            )
        except RedisError as error:
            self._on_error("lookup", error)
            self._stats.record_error(SEMANTIC_ANSWER_CACHE, 1)
            return None

        self._breaker.record_success()
        best = _best_match(result.docs)
        hit = best is not None and best.similarity >= self._threshold
        self._stats.record_lookup(SEMANTIC_ANSWER_CACHE, int(hit), int(not hit), time.perf_counter() - started)
        return best if hit else None

    def store(self, scope: str, vector: Sequence[float], question: str, answer: str) -> None:
        """Save an answer; the same question in the same scope is overwritten."""
        if self._unsupported or not self._breaker.allow():
            return
        dimensions = len(vector)
        scope_hash = _hash(scope)
        key = f"{self._key_prefix}:sem:{dimensions}:{_hash(scope_hash + chr(10) + question)[:40]}"
        try:
            self._ensure_index(dimensions)
            pipe = self._client_factory().pipeline(transaction=False)
            pipe.hset(
                key,
                mapping={
                    "scope": scope_hash,
                    "question": question,
                    "answer": answer,
                    "embedding": _to_bytes(vector),
                },
            )
            pipe.expire(key, self._ttl_seconds)
            pipe.execute()
        except RedisError as error:
            self._on_error("store", error)
            return
        self._breaker.record_success()
        self._stats.record_writes(SEMANTIC_ANSWER_CACHE, 1)

    def _index_name(self, dimensions: int) -> str:
        return f"{self._key_prefix}:sem-idx:{dimensions}"

    def _ensure_index(self, dimensions: int) -> None:
        """Create the vector index for this size once."""
        if dimensions in self._indexed_dimensions:
            return
        with self._lock:
            if dimensions in self._indexed_dimensions:
                return
            schema = (
                TagField("scope"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": dimensions, "DISTANCE_METRIC": "COSINE"},
                ),
            )
            definition = IndexDefinition(
                prefix=[f"{self._key_prefix}:sem:{dimensions}:"], index_type=IndexType.HASH
            )
            try:
                self._client_factory().ft(self._index_name(dimensions)).create_index(
                    schema, definition=definition
                )
                logger.info(f"Created semantic cache index {self._index_name(dimensions)}")
            except ResponseError as error:
                if "already exists" not in str(error).lower():
                    raise
            self._indexed_dimensions.add(dimensions)

    def _on_error(self, operation: str, error: RedisError) -> None:
        if isinstance(error, ResponseError) and "unknown command" in str(error).lower():
            # Redis has no vector search.
            self._unsupported = True
            logger.error(
                "Semantic cache disabled: this Redis has no vector search "
                "(use Redis 8+ or Redis Stack)"
            )
            return
        if isinstance(error, ResponseError):
            # Redis rejected the command.
            logger.error(f"semantic cache {operation} rejected by Redis: {error}")
            return
        self._breaker.record_failure()
        logger.warning(f"semantic cache {operation} failed ({type(error).__name__}); skipped")


def _best_match(docs) -> Optional[CachedAnswer]:
    if not docs:
        return None
    doc = docs[0]
    return CachedAnswer(
        question=_text(doc.question),
        answer=_text(doc.answer),
        # Redis returns distance, not similarity.
        similarity=round(1.0 - float(_text(doc.distance)), 4),
    )


def _to_bytes(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
