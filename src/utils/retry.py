"""Retry utilities with exponential backoff for robust API calls."""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial attempt)
        base_delay: Base delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retriable_exceptions: Tuple of exception types that should trigger retries
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,  # Includes network errors
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")

        # Additional validation for sensible limits
        if self.max_attempts > 10:
            raise ValueError("max_attempts should not exceed 10 for practical purposes")
        if self.max_delay > 300:  # 5 minutes
            raise ValueError("max_delay should not exceed 300 seconds for practical purposes")

    def calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter and ceiling.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Calculated delay in seconds
        """
        return calculate_delay(
            attempt, self.base_delay, self.max_delay, self.exponential_base, self.jitter
        )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made
        last_exception: The final exception that caused the retry to fail
        total_delay: Total time spent in delays
    """

    def __init__(self, attempts: int, last_exception: Exception, total_delay: float) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        self.total_delay = total_delay
        super().__init__(
            f"Retry exhausted after {attempts} attempts over {total_delay:.2f}s. "
            f"Last error: {last_exception}"
        )


def calculate_delay(
    attempt: int, base_delay: float, max_delay: float, exponential_base: float, jitter: bool = True
) -> float:
    """Calculate delay for a given retry attempt with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Calculated delay in seconds, always between 0 and max_delay
    """
    if attempt == 0:
        return 0.0

    # Calculate exponential backoff with ceiling
    base_backoff = min(
        base_delay * (exponential_base ** (attempt - 1)), max_delay
    )  # Ensure we never exceed max_delay

    # Add jitter if enabled
    if jitter:
        # Add ±25% random jitter
        jitter_range = base_backoff * 0.25
        base_backoff += random.uniform(-jitter_range, jitter_range)

    # Final ceiling check and ensure non-negative
    return max(0, min(base_backoff, max_delay))


def is_retriable_exception(
    exception: Exception, retriable_exceptions: Tuple[Type[Exception], ...]
) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exception: The exception to check
        retriable_exceptions: Tuple of exception types that are retriable

    Returns:
        True if exception should trigger retry, False otherwise
    """
    # Check for specific HTTP status codes if available
    if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        status_code = exception.response.status_code
        # Retry on 5xx server errors and specific 4xx errors
        retriable_status_codes = {408, 429, 500, 502, 503, 504}
        if status_code in retriable_status_codes:
            return True
        # Don't retry on other 4xx client errors
        if 400 <= status_code < 500:
            return False

    # Check if exception type is in retriable list
    return isinstance(exception, retriable_exceptions)


def retry_sync(
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    jitter: Optional[bool] = None,
    retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator for synchronous functions with retry logic.

    Args:
        config: RetryConfig instance (overrides individual parameters)
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retriable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_sync(max_attempts=3, base_delay=1.0)
        def make_api_call():
            # API call that might fail
            pass
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 1.0,
            max_delay=max_delay or 60.0,
            exponential_base=exponential_base or 2.0,
            jitter=jitter if jitter is not None else True,
            retriable_exceptions=retriable_exceptions or RetryConfig().retriable_exceptions,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            total_delay = 0.0

            for attempt in range(config.max_attempts):
                try:
                    if attempt > 0:
                        logger.info(
                            f"Retry attempt {attempt + 1}/{config.max_attempts} for {func.__name__}"
                        )
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not is_retriable_exception(e, config.retriable_exceptions):
                        logger.error(f"Non-retriable exception in {func.__name__}: {e}")
                        raise

                    # Check if we have more attempts left
                    if attempt + 1 >= config.max_attempts:
                        logger.error(f"All retry attempts exhausted for {func.__name__}: {e}")
                        break

                    # Calculate and apply delay
                    delay = calculate_delay(
                        attempt + 1,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )

                    total_delay += delay

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # All attempts failed
            raise RetryExhaustedError(
                config.max_attempts, last_exception or Exception("Unknown error"), total_delay
            )

        return wrapper  # type: ignore

    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    jitter: Optional[bool] = None,
    retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[AsyncF], AsyncF]:
    """Decorator for asynchronous functions with retry logic.

    Args:
        config: RetryConfig instance (overrides individual parameters)
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retriable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated async function with retry logic

    Example:
        @retry_async(max_attempts=3, base_delay=1.0)
        async def make_api_call():
            # Async API call that might fail
            pass
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 1.0,
            max_delay=max_delay or 60.0,
            exponential_base=exponential_base or 2.0,
            jitter=jitter if jitter is not None else True,
            retriable_exceptions=retriable_exceptions or RetryConfig().retriable_exceptions,
        )

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            total_delay = 0.0

            for attempt in range(config.max_attempts):
                try:
                    if attempt > 0:
                        logger.info(
                            f"Retry attempt {attempt + 1}/{config.max_attempts} for {func.__name__}"
                        )
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not is_retriable_exception(e, config.retriable_exceptions):
                        logger.error(f"Non-retriable exception in {func.__name__}: {e}")
                        raise

                    # Check if we have more attempts left
                    if attempt + 1 >= config.max_attempts:
                        logger.error(f"All retry attempts exhausted for {func.__name__}: {e}")
                        break

                    # Calculate and apply delay
                    delay = calculate_delay(
                        attempt + 1,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )

                    total_delay += delay

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # All attempts failed
            raise RetryExhaustedError(
                config.max_attempts, last_exception or Exception("Unknown error"), total_delay
            )

        return wrapper  # type: ignore

    return decorator


# Convenience functions for common retry scenarios
def retry_on_network_error(
    max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
) -> Callable[[F], F]:
    """Convenience decorator for network-related errors.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorator configured for network errors
    """
    network_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    return retry_sync(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retriable_exceptions=network_exceptions,
    )


def retry_on_network_error_async(
    max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
) -> Callable[[AsyncF], AsyncF]:
    """Convenience decorator for network-related errors (async version).

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Async decorator configured for network errors
    """
    network_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    return retry_async(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retriable_exceptions=network_exceptions,
    )


class RetryBudget:
    """Manages retry budget across operations to prevent excessive retries.

    This class implements a sliding window approach to limit the number of
    retry attempts within a given time window, preventing retry storms.
    """

    def __init__(self, max_budget: int = 100, window_seconds: int = 60) -> None:
        """Initialize retry budget manager.

        Args:
            max_budget: Maximum number of retries allowed within the time window
            window_seconds: Time window in seconds for budget tracking
        """
        self.max_budget = max_budget
        self.window_seconds = window_seconds
        self.attempts: List[float] = []
        self._lock = threading.Lock()

    def can_retry(self) -> bool:
        """Check if retry budget allows another attempt.

        Returns:
            True if retry is allowed within budget, False otherwise
        """
        with self._lock:
            now = time.time()
            # Remove old attempts outside the time window
            self.attempts = [t for t in self.attempts if now - t < self.window_seconds]

            if len(self.attempts) < self.max_budget:
                self.attempts.append(now)
                return True
            return False

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status information.

        Returns:
            Dictionary containing budget status details
        """
        with self._lock:
            now = time.time()
            # Clean up old attempts for accurate count
            self.attempts = [t for t in self.attempts if now - t < self.window_seconds]

            return {
                "used_budget": len(self.attempts),
                "max_budget": self.max_budget,
                "remaining_budget": self.max_budget - len(self.attempts),
                "window_seconds": self.window_seconds,
                "budget_available": len(self.attempts) < self.max_budget,
            }
