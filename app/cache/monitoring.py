import asyncio
import json

from loguru import logger
from redis.exceptions import RedisError

from app.cache.client import get_redis_client
from app.cache.stats import cache_stats
from app.cache.store import get_circuit_breaker
from app.settings import config_settings

# Redis INFO fields we report.
_REDIS_INFO_FIELDS = (
    "redis_version",
    "connected_clients",
    "used_memory_human",
    "maxmemory_human",
    "maxmemory_policy",
    "keyspace_hits",
    "keyspace_misses",
    "evicted_keys",
    "expired_keys",
)


def redis_server_info() -> dict:
    """Redis server stats (blocking)."""
    try:
        info = get_redis_client().info()
    except RedisError as error:
        return {"reachable": False, "error": type(error).__name__}

    summary = {"reachable": True, **{field: info.get(field) for field in _REDIS_INFO_FIELDS}}
    summary["keyspace"] = {name: value for name, value in info.items() if name.startswith("db")}
    return summary


def cache_report() -> dict:
    """Cache counters plus Redis stats (blocking)."""
    if not config_settings.CACHE_ENABLED:
        return {"enabled": False}
    return {
        "enabled": True,
        "circuit_open": get_circuit_breaker().is_open,
        "caches": cache_stats.snapshot(),
        "redis": redis_server_info(),
    }


async def log_cache_stats_periodically(interval_seconds: int) -> None:
    """Log one JSON cache report per interval."""
    while True:
        await asyncio.sleep(interval_seconds)
        report = await asyncio.to_thread(cache_report)
        logger.info(f"cache stats {json.dumps(report, default=str)}")
