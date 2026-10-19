import threading
from dataclasses import dataclass


@dataclass
class _Counters:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    bypassed: int = 0
    writes: int = 0
    lookups: int = 0
    lookup_seconds: float = 0.0


class CacheStats:
    """Thread-safe hit/miss/error counters per cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, _Counters] = {}

    def _get(self, cache: str) -> _Counters:
        return self._counters.setdefault(cache, _Counters())

    def record_lookup(self, cache: str, hits: int, misses: int, seconds: float) -> None:
        with self._lock:
            counters = self._get(cache)
            counters.hits += hits
            counters.misses += misses
            counters.lookups += 1
            counters.lookup_seconds += seconds

    def record_error(self, cache: str, keys: int) -> None:
        with self._lock:
            counters = self._get(cache)
            counters.errors += 1
            counters.misses += keys

    def record_bypass(self, cache: str, keys: int) -> None:
        with self._lock:
            counters = self._get(cache)
            counters.bypassed += keys
            counters.misses += keys

    def record_writes(self, cache: str, keys: int) -> None:
        with self._lock:
            self._get(cache).writes += keys

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {name: _summary(counters) for name, counters in self._counters.items()}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


def _summary(counters: _Counters) -> dict:
    total = counters.hits + counters.misses
    return {
        "hits": counters.hits,
        "misses": counters.misses,
        "hit_rate": round(counters.hits / total, 4) if total else None,
        "errors": counters.errors,
        "bypassed": counters.bypassed,
        "writes": counters.writes,
        "avg_lookup_ms": round(1000 * counters.lookup_seconds / counters.lookups, 2)
        if counters.lookups
        else None,
    }


# Shared by every cache.
cache_stats = CacheStats()
