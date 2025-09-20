"""Error Coordination Module - Production-Ready Resilience Framework.

This module provides comprehensive error handling, circuit breakers,
retry strategies, and cascade prevention for the audio extraction pipeline.

Quick Start:
-----------
from error_coordination import with_error_coordination, error_coordinator

@with_error_coordination(
    operation="transcription",
    circuit_breaker=True,
    retry=True
)
def transcribe_audio(path: Path) -> Optional[TranscriptionResult]:
    # Your transcription logic
    pass

# Register custom error handlers
error_coordinator.register_error_handler(
    ConnectionError,
    lambda e: logger.error(f"Connection failed: {e}")
)

# Register recovery strategies  
error_coordinator.register_recovery_strategy(
    "transcription",
    lambda: use_fallback_provider()
)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Type imports for documentation
try:
    from ..models.transcription import TranscriptionResult
except ImportError:
    TranscriptionResult = None  # type: ignore

from .config import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorCoordinator,
    ErrorMetrics,
    ErrorSeverity,
    RetryConfig,
    RetryStrategy,
    aggressive_retry,
    conservative_retry,
    default_retry,
    error_coordinator,
    with_error_coordination,
)

__version__ = "1.0.0"

__all__ = [
    # Main coordinator
    "error_coordinator",
    "ErrorCoordinator",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    
    # Retry strategies
    "RetryStrategy",
    "RetryConfig",
    "default_retry",
    "aggressive_retry", 
    "conservative_retry",
    
    # Decorators and utilities
    "with_error_coordination",
    "ErrorSeverity",
    "ErrorMetrics",
]
