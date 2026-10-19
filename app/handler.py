import asyncio
import inspect
import time
from functools import wraps
from typing import Callable, Optional, Type

from loguru import logger


def retry_with_custom_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 20.0,
    exception: tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[BaseException, int], None]] = None,
):
    """Retry a sync or async function with exponential backoff."""

    def delay_for(attempt: int) -> float:
        return min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)

    def decorator(func: Callable) -> Callable:
        name = getattr(func, "__name__", repr(func))

        def _handle_failure(error: BaseException, attempt: int) -> float:
            if attempt >= max_retries:
                logger.error(f"{name} failed after {max_retries} attempts: {error}")
                raise error
            delay = delay_for(attempt)
            if on_retry:
                on_retry(error, attempt)
            logger.warning(
                f"Retrying {name} in {delay:.1f}s (attempt {attempt}/{max_retries}): {error}"
            )
            return delay

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                for attempt in range(1, max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exception as error:
                        await asyncio.sleep(_handle_failure(error, attempt))

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception as error:
                    time.sleep(_handle_failure(error, attempt))

        return sync_wrapper

    return decorator
