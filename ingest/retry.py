"""SRAG Retry Mechanism — Phase 2 implementation (DD-12).

Provides exponential backoff retry for transient failures during ingestion,
embedding, and LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Type alias for callable return types.
R = TypeVar("R")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, attempts: int = 0, last_error: Exception | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception]] | None = None,
) -> Callable[[Callable[R, Any]], Callable[..., R]]:
    """Decorator that retries a function with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (not counting the initial attempt).
    base_delay : float
        Initial delay in seconds between retries.
    max_delay : float
        Maximum delay cap in seconds.
    exponential_base : float
        Base for exponential backoff calculation.
    jitter : bool
        Whether to add random jitter to prevent thundering herd.
    retry_on : tuple[type[Exception]] | None
        Tuple of exception types to retry on. ``None`` retries all exceptions.

    Returns
    -------
    Callable[[Callable[R, Any]], Callable[..., R]]
        The wrapped function.

    Raises
    ------
    RetryExhaustedError
        If all retries are exhausted. The original exception is attached
        as ``last_error``.
    """
    def decorator(func: Callable[R, Any]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            last_exc: Exception | None = None

            for attempt in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if retry_on is not None and not isinstance(exc, retry_on):
                        raise  # Re-raise non-matching exceptions immediately.

                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random())  # ±50% jitter.
                        logger.warning(
                            "Attempt %d/%d failed for '%s': %s. Retrying in %.2fs.",
                            attempt + 1,
                            max_retries + 1,
                            func.__qualname__,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts exhausted for '%s'. Last error: %s",
                            max_retries + 1,
                            func.__qualname__,
                            last_exc,
                        )

            raise RetryExhaustedError(
                f"All {max_retries + 1} attempts failed for '{func.__qualname__}'",
                attempts=max_retries + 1,
                last_error=last_exc,
            ) from last_exc

        return wrapper

    return decorator


async def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Async version of :func:`retry`.

    Parameters are identical to :func:`retry`. Returns an async decorator.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None

            for attempt in range(1 + max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if retry_on is not None and not isinstance(exc, retry_on):
                        raise

                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random())
                        logger.warning(
                            "Attempt %d/%d failed for '%s': %s. Retrying in %.2fs.",
                            attempt + 1,
                            max_retries + 1,
                            func.__qualname__,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts exhausted for '%s'. Last error: %s",
                            max_retries + 1,
                            func.__qualname__,
                            last_exc,
                        )

            raise RetryExhaustedError(
                f"All {max_retries + 1} attempts failed for '{func.__qualname__}'",
                attempts=max_retries + 1,
                last_error=last_exc,
            ) from last_exc

        return wrapper

    return decorator


class RetryPolicy:
    """Configurable retry policy for ingestion stages.

    Example
    -------
    >>> policy = RetryPolicy(max_retries=5, base_delay=0.5)
    >>> @policy.wrap
    ... def fragile_operation():
    ...     pass
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """Initialize the retry policy.

        Parameters
        ----------
        max_retries : int
        base_delay : float
        max_delay : float
        exponential_base : float
        jitter : bool
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def wrap(self, func: Callable[R, Any]) -> Callable[..., R]:
        """Apply retry decoration to *func*."""
        return retry(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
        )(func)

    def async_wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply async retry decoration to *func*."""
        # Return a decorator that will be awaited.
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            return asyncio_retry(
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
                exponential_base=self.exponential_base,
                jitter=self.jitter,
            )(f)

        # We need to return the wrapper directly.
        return decorator


__all__ = [
    "RetryExhaustedError",
    "retry",
    "async_retry",
    "RetryPolicy",
]