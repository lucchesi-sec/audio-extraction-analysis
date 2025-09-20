"""Abstract base class for transcription service providers."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

from ..models.transcription import TranscriptionResult
from ..utils.retry import RetryConfig, retry_async

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before attempting recovery
        expected_exception_types: Exception types that count as failures
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception_types: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, failure_count: int, last_failure_time: float) -> None:
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        super().__init__(message)


class CircuitBreakerMixin:
    """Mixin class that provides circuit breaker functionality for providers.

    This mixin can be used by any transcription provider to add circuit breaker
    pattern for preventing cascading failures when external services are unavailable.
    """

    def __init__(self, circuit_config: Optional[CircuitBreakerConfig] = None) -> None:
        """Initialize circuit breaker state.

        Args:
            circuit_config: Configuration for circuit breaker behavior
        """
        self._circuit_config = circuit_config or CircuitBreakerConfig()
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = Lock()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset to half-open.

        Returns:
            True if enough time has passed to attempt reset
        """
        with self._lock:  # Add lock protection for thread safety
            return (
                self._circuit_state == CircuitState.OPEN
                and time.time() - self._last_failure_time >= self._circuit_config.recovery_timeout
            )

    def _record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._failure_count = 0
            self._circuit_state = CircuitState.CLOSED

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed operation.

        Args:
            exception: The exception that caused the failure
        """
        # Only count expected exception types as failures
        if not isinstance(exception, self._circuit_config.expected_exception_types):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self._circuit_config.failure_threshold:
                self._circuit_state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures. "
                    f"Will attempt recovery in {self._circuit_config.recovery_timeout}s"
                )

    def _check_circuit_state(self) -> None:
        """Check circuit state and raise exception if open.

        Raises:
            CircuitBreakerError: If circuit is open and recovery timeout hasn't passed
        """
        with self._lock:
            if self._circuit_state == CircuitState.OPEN:
                # Check reset condition within the same lock to avoid race condition
                if time.time() - self._last_failure_time >= self._circuit_config.recovery_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is open. {self._failure_count} consecutive failures. "
                        f"Will retry after {self._circuit_config.recovery_timeout}s",
                        self._failure_count,
                        self._last_failure_time,
                    )

    def circuit_breaker_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        self._check_circuit_state()

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def circuit_breaker_call_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        self._check_circuit_state()

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def get_circuit_state(self) -> Dict[str, Union[str, int, float]]:
        """Get current circuit breaker state information.

        Returns:
            Dictionary containing circuit state information
        """
        with self._lock:
            return {
                "state": self._circuit_state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self._circuit_config.failure_threshold,
                "last_failure_time": self._last_failure_time,
                "recovery_timeout": self._circuit_config.recovery_timeout,
                "time_until_retry": (
                    max(
                        0,
                        self._last_failure_time
                        + self._circuit_config.recovery_timeout
                        - time.time(),
                    )
                    if self._circuit_state == CircuitState.OPEN
                    else 0
                ),
            }


class BaseTranscriptionProvider(ABC, CircuitBreakerMixin):
    """Abstract base class for all transcription service providers.

    This class defines the common interface that all transcription providers
    must implement, ensuring consistency across different services like
    Deepgram, ElevenLabs, etc.

    Includes circuit breaker functionality to prevent cascading failures.
    """

    # Default configurations for all providers
    DEFAULT_RETRY_CONFIG = RetryConfig(
        max_attempts=3, base_delay=1.0, exponential_base=2, max_delay=30.0, jitter=True
    )

    DEFAULT_CIRCUIT_CONFIG = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception_types=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the transcription provider.

        Args:
            api_key: Optional API key for the service
            circuit_config: Circuit breaker configuration (uses DEFAULT_CIRCUIT_CONFIG if None)
            retry_config: Retry configuration (uses DEFAULT_RETRY_CONFIG if None)
        """
        self.api_key = api_key
        self._retry_config = retry_config or self.DEFAULT_RETRY_CONFIG

        # Initialize circuit breaker with default config if not provided
        CircuitBreakerMixin.__init__(self, circuit_config or self.DEFAULT_CIRCUIT_CONFIG)

    @abstractmethod
    async def _transcribe_impl(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Internal implementation of transcription.

        This method should contain the actual transcription logic
        without retry or circuit breaker handling.

        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription (e.g., 'en', 'es')

        Returns:
            TranscriptionResult object with all available features, or None if failed
        """
        pass

    async def transcribe_async(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio file asynchronously with retry and circuit breaker.

        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription (e.g., 'en', 'es')

        Returns:
            TranscriptionResult object with all available features, or None if failed
        """

        @retry_async(config=self._retry_config)
        async def _transcribe_with_retry():
            return await self._transcribe_impl(audio_file_path, language)

        try:
            return await self.circuit_breaker_call_async(_transcribe_with_retry)
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker prevented transcription: {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def transcribe(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio file synchronously with retry and circuit breaker.

        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription (e.g., 'en', 'es')

        Returns:
            TranscriptionResult object with all available features, or None if failed
        """
        return asyncio.run(self.transcribe_async(audio_file_path, language))

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate that the provider is properly configured.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this transcription provider.

        Returns:
            Human-readable name of the provider (e.g., 'Deepgram Nova 3', 'ElevenLabs')
        """
        pass

    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Get list of features supported by this provider.

        Returns:
            List of feature names like 'speaker_diarization', 'topic_detection',
            'sentiment_analysis', 'timestamps', etc.
        """
        pass

    @abstractmethod
    async def health_check_async(self) -> Dict[str, Any]:
        """Perform asynchronous health check for the provider.

        This should verify API connectivity, authentication, and service availability.

        Returns:
            Dictionary containing health check results:
            {
                "healthy": bool,
                "status": str,
                "response_time_ms": float,
                "details": dict
            }
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """Perform synchronous health check for the provider.

        Returns:
            Dictionary containing health check results
        """
        return asyncio.run(self.health_check_async())

    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported, False otherwise
        """
        return feature in self.get_supported_features()

    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration for this provider.

        Returns:
            RetryConfig instance
        """
        return self._retry_config

    def update_retry_config(self, config: RetryConfig) -> None:
        """Update retry configuration for this provider.

        Args:
            config: New retry configuration
        """
        self._retry_config = config

    # ---------------------- Progress Helper ----------------------
    def _report_progress(self, callback: Optional[Callable], completed: int, total: int) -> None:
        """Helper to report progress if a callback is provided.

        Args:
            callback: Optional callable taking (completed, total)
            completed: Completed units
            total: Total units
        """
        if callback:
            try:
                callback(completed, total)
            except Exception:
                # Never let a progress callback break the provider
                pass
