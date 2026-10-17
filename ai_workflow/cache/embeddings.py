from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.storage import RedisStore
from langchain_core.embeddings import Embeddings

from ai_workflow.cache.client import get_redis_client
from ai_workflow.cache.stats import cache_stats
from ai_workflow.cache.store import MonitoredStore, get_circuit_breaker
from ai_workflow.settings import config_settings

QUERY_EMBEDDING_CACHE = "embedding_query"
DOCUMENT_EMBEDDING_CACHE = "embedding_document"


def _monitored_redis_store(cache_name: str, kind: str, ttl_seconds: int) -> MonitoredStore:
    redis_store = RedisStore(
        client=get_redis_client(),
        ttl=ttl_seconds,
        namespace=f"{config_settings.CACHE_KEY_PREFIX}:emb:{kind}",
    )
    return MonitoredStore(redis_store, cache_name, cache_stats, get_circuit_breaker())


def build_cached_embeddings(underlying: Embeddings, model_id: str) -> Embeddings:
    """Wrap ``underlying`` so repeated texts are served from Redis."""
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        document_embedding_cache=_monitored_redis_store(
            DOCUMENT_EMBEDDING_CACHE, "document", config_settings.CACHE_DOCUMENT_EMBEDDING_TTL_SECONDS
        ),
        query_embedding_cache=_monitored_redis_store(
            QUERY_EMBEDDING_CACHE, "query", config_settings.CACHE_QUERY_EMBEDDING_TTL_SECONDS
        ),
        namespace=f"{model_id}:",
        key_encoder="sha256",
    )
