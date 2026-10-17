import asyncio
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

from langsmith import traceable
from loguru import logger

from ai_workflow.cache.client import get_redis_client
from ai_workflow.cache.semantic import CachedAnswer, SemanticAnswerStore
from ai_workflow.cache.stats import cache_stats
from ai_workflow.cache.store import get_circuit_breaker
from ai_workflow.cache.versions import DocumentVersions
from ai_workflow.retrieval.prompts import initialise_doc_search_prompt_template
from ai_workflow.retrieval.question_rules import is_standalone, normalize_question
from ai_workflow.settings import config_settings
from ai_workflow.utils import chat_model_id, embedding_model_id, get_embeddings
from ai_workflow.vector_db.sanitization import sanitize_index_name, sanitize_namespace

# Start of the prompt's fixed "not found" reply.
_NOT_FOUND_REPLY = "i don't have that specific information"


@dataclass(frozen=True)
class CacheProbe:
    """Lookup result; reused to store the answer on a miss."""

    scope: str
    question: str
    vector: list[float]
    hit: Optional[CachedAnswer]


def is_cacheable_answer(answer: str) -> bool:
    text = answer.strip().casefold().replace("’", "'")
    return bool(text) and not text.startswith(_NOT_FOUND_REPLY)


def documents_scope(index_name: str, namespace: str) -> str:
    """Scope string matching Weaviate's naming."""
    return f"{sanitize_index_name(index_name)}/{sanitize_namespace(namespace)}"


@lru_cache(maxsize=1)
def _answer_prompt_fingerprint() -> str:
    template = initialise_doc_search_prompt_template().template
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def answer_scope(documents: str, documents_version: str, language: str) -> str:
    return "|".join(
        (
            documents,
            documents_version,
            language.strip().casefold(),
            chat_model_id(),
            _answer_prompt_fingerprint(),
            embedding_model_id(),
        )
    )


class AnswerCache:
    def __init__(
        self,
        store: SemanticAnswerStore,
        versions: DocumentVersions,
        embed_query: Callable[[str], list[float]],
    ) -> None:
        self._store = store
        self._versions = versions
        self._embed_query = embed_query

    @traceable(
        name="semantic_cache_lookup",
        run_type="tool",
        process_inputs=lambda inputs: {
            "question": inputs["question"],
            "has_history": inputs["has_history"],
            "index_name": inputs["index_name"],
            "namespace": inputs["namespace"],
        },
        process_outputs=lambda probe: {
            "eligible": probe is not None,
            "hit": bool(probe and probe.hit),
            "similarity": probe.hit.similarity if probe and probe.hit else None,
            "matched_question": probe.hit.question if probe and probe.hit else None,
        },
    )
    async def probe(
        self, question: str, has_history: bool, language: str, index_name: str, namespace: str
    ) -> Optional[CacheProbe]:
        """Look up the question; None if not eligible or the cache is unavailable."""
        if not is_standalone(question, has_history):
            return None
        normalized = normalize_question(question)
        if not normalized:
            return None
        try:
            documents = documents_scope(index_name, namespace)
        except ValueError:
            return None

        version = await asyncio.to_thread(self._versions.current, documents)
        if version is None:
            return None
        try:
            vector = await asyncio.to_thread(self._embed_query, normalized)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"semantic cache skipped: question embedding failed ({error})")
            return None

        scope = answer_scope(documents, version, language)
        try:
            hit = await asyncio.to_thread(self._store.lookup, scope, vector)
        except Exception:  # noqa: BLE001
            logger.exception("semantic cache lookup failed")
            return None
        return CacheProbe(scope=scope, question=normalized, vector=vector, hit=hit)

    async def save(self, probe: CacheProbe, answer: str) -> None:
        if probe.hit is not None or not is_cacheable_answer(answer):
            return
        try:
            await asyncio.to_thread(self._store.store, probe.scope, probe.vector, probe.question, answer)
        except Exception:  # noqa: BLE001
            logger.exception("semantic cache store failed")


@lru_cache(maxsize=1)
def get_document_versions() -> DocumentVersions:
    return DocumentVersions(
        get_redis_client,
        get_circuit_breaker(),
        key_prefix=config_settings.CACHE_KEY_PREFIX,
        ttl_seconds=config_settings.DOCUMENT_VERSION_TTL_SECONDS,
    )


@lru_cache(maxsize=1)
def get_answer_cache() -> Optional[AnswerCache]:
    """The answer cache, or None when SEMANTIC_CACHE_ENABLED is off."""
    if not config_settings.SEMANTIC_CACHE_ENABLED:
        return None
    store = SemanticAnswerStore(
        get_redis_client,
        cache_stats,
        get_circuit_breaker(),
        key_prefix=config_settings.CACHE_KEY_PREFIX,
        threshold=config_settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds=config_settings.SEMANTIC_CACHE_TTL_SECONDS,
    )
    return AnswerCache(store, get_document_versions(), lambda text: get_embeddings().embed_query(text))


async def invalidate_cached_answers(index_name: str, namespace: str) -> None:
    """Invalidate cached answers after documents change (ingest or delete)."""
    if not config_settings.CACHE_ENABLED:
        return
    try:
        documents = documents_scope(index_name, namespace)
    except ValueError:
        return
    if await asyncio.to_thread(get_document_versions().bump, documents):
        logger.info(f"Cached answers invalidated for {documents}")