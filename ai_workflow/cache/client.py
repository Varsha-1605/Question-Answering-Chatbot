from functools import lru_cache

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from ai_workflow.settings import config_settings


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """Shared sync Redis client with short timeouts and no retries (fails fast)."""
    return redis.Redis.from_url(
        config_settings.REDIS_URL.get_secret_value(),
        socket_connect_timeout=config_settings.REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=config_settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        retry=Retry(NoBackoff(), 0),
        health_check_interval=30,
    )


def close_redis_client() -> None:
    """Close the client if it was opened."""
    if get_redis_client.cache_info().currsize:
        get_redis_client().close()
        get_redis_client.cache_clear()
