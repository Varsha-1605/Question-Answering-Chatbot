import threading
import time
from functools import lru_cache
from typing import Callable, Iterator, Optional, Sequence

from langchain_core.stores import ByteStore
from loguru import logger
from redis.exceptions import RedisError

from ai_workflow.cache.stats import CacheStats
from ai_workflow.settings import config_settings


class CircuitBreaker:
    """Skips Redis calls for a while after a failure."""

    def __init__(self, open_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._open_seconds = open_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return self._clock() < self._open_until

    def allow(self) -> bool:
        return not self.is_open

    def record_failure(self) -> None:
        with self._lock:
            self._open_until = self._clock() + self._open_seconds

    def record_success(self) -> None:
        with self._lock:
            self._open_until = 0.0


@lru_cache(maxsize=1)
def get_circuit_breaker() -> CircuitBreaker:
    """One shared breaker for all caches."""
    return CircuitBreaker(open_seconds=config_settings.CACHE_CIRCUIT_OPEN_SECONDS)


class MonitoredStore(ByteStore):
    def __init__(self, inner: ByteStore, cache_name: str, stats: CacheStats, breaker: CircuitBreaker) -> None:
        self._inner = inner
        self._cache_name = cache_name
        self._stats = stats
        self._breaker = breaker

    def mget(self, keys: Sequence[str]) -> list[Optional[bytes]]:
        if not keys:
            return []
        if not self._breaker.allow():
            self._stats.record_bypass(self._cache_name, len(keys))
            return [None] * len(keys)

        started = time.perf_counter()
        try:
            values = self._inner.mget(keys)
        except RedisError as error:
            self._on_error("read", error)
            self._stats.record_error(self._cache_name, len(keys))
            return [None] * len(keys)

        self._breaker.record_success()
        hits = sum(value is not None for value in values)
        self._stats.record_lookup(self._cache_name, hits, len(keys) - hits, time.perf_counter() - started)
        logger.debug(f"cache {self._cache_name}: {hits}/{len(keys)} hits")
        return values

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        if not key_value_pairs or not self._breaker.allow():
            return
        try:
            self._inner.mset(key_value_pairs)
        except RedisError as error:
            self._on_error("write", error)
            return
        self._breaker.record_success()
        self._stats.record_writes(self._cache_name, len(key_value_pairs))

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys or not self._breaker.allow():
            return
        try:
            self._inner.mdelete(keys)
        except RedisError as error:
            self._on_error("delete", error)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        yield from self._inner.yield_keys(prefix=prefix)

    def _on_error(self, operation: str, error: RedisError) -> None:
        self._breaker.record_failure()
        logger.warning(
            f"cache {self._cache_name}: {operation} failed ({type(error).__name__}); "
            f"Redis skipped for {config_settings.CACHE_CIRCUIT_OPEN_SECONDS:.0f}s"
        )
