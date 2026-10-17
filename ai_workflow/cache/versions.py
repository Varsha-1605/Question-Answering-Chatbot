import uuid
from typing import Callable, Optional

import redis
from loguru import logger
from redis.exceptions import RedisError

from ai_workflow.cache.store import CircuitBreaker


class DocumentVersions:
    def __init__(
        self,
        client_factory: Callable[[], redis.Redis],
        breaker: CircuitBreaker,
        *,
        key_prefix: str,
        ttl_seconds: int,
    ) -> None:
        self._client_factory = client_factory
        self._breaker = breaker
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    def current(self, scope: str) -> Optional[str]:
        """Return the scope's token, or None if Redis is unavailable."""
        if not self._breaker.allow():
            return None
        key = self._key(scope)
        try:
            client = self._client_factory()
            # Set only if missing, then read.
            client.set(key, uuid.uuid4().hex, nx=True, ex=self._ttl_seconds)
            token = client.get(key)
        except RedisError as error:
            self._breaker.record_failure()
            logger.warning(f"document version read failed ({type(error).__name__}); cache skipped")
            return None
        self._breaker.record_success()
        return token.decode() if isinstance(token, bytes) else token

    def bump(self, scope: str) -> bool:
        """Replace the scope's token, even if the breaker is open."""
        try:
            self._client_factory().set(self._key(scope), uuid.uuid4().hex, ex=self._ttl_seconds)
        except RedisError as error:
            logger.error(
                f"Could not invalidate cached answers for {scope!r} ({type(error).__name__}); "
                "they can be stale until their TTL ends"
            )
            return False
        return True

    def _key(self, scope: str) -> str:
        return f"{self._key_prefix}:docver:{scope}"
