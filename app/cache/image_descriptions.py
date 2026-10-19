from typing import Optional

from langchain_community.storage import RedisStore
from langchain_core.stores import ByteStore

from app.cache.client import get_redis_client
from app.cache.stats import cache_stats
from app.cache.store import MonitoredStore, get_circuit_breaker
from app.settings import config_settings

IMAGE_DESCRIPTION_CACHE = "image_description"


def build_image_description_store() -> Optional[ByteStore]:
    """The monitored Redis store, or None when caching is off."""
    if not config_settings.CACHE_ENABLED:
        return None
    redis_store = RedisStore(
        client=get_redis_client(),
        ttl=config_settings.CACHE_IMAGE_DESCRIPTION_TTL_SECONDS,
        namespace=f"{config_settings.CACHE_KEY_PREFIX}:img",
    )
    return MonitoredStore(redis_store, IMAGE_DESCRIPTION_CACHE, cache_stats, get_circuit_breaker())
