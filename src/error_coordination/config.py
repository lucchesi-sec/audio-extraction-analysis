"""Error Coordination and Resilience Configuration.

This module implements comprehensive error coordination, circuit breakers,
retry strategies, and failure recovery mechanisms for production deployment.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting calls  
    HALF_OPEN = "half_open"  # Testing recovery


class ErrorSeverity(Enum):
    """Error severity levels for coordination."""
    LOW = 1      # Recoverable, retry immediately
    MEDIUM = 2   # Recoverable with backoff
    HIGH = 3     # Circuit breaker trigger
    CRITICAL = 4  # Cascade prevention needed


@dataclass
class ErrorMetrics:
    """Track error metrics for coordination."""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    last_error: Optional[datetime] = None
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    mttr_seconds: float = 0.0  # Mean Time To Recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_attempts: int = 3
    error_severity_threshold: ErrorSeverity = ErrorSeverity.HIGH


@dataclass  
class RetryConfig:
    """Retry strategy configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ErrorCoordinator:
    """Central error coordination and resilience management."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_metrics: Dict[str, ErrorMetrics] = {}
        self.error_handlers: Dict[type, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self._error_history = deque(maxlen=1000)
        
    def register_error_handler(
        self,
        error_type: type,
        handler: Callable[[Exception], Any]
    ):
        """Register custom error handler for specific exception types."""
        self.error_handlers[error_type] = handler
        logger.info(f"Registered handler for {error_type.__name__}")
    
    def register_recovery_strategy(
        self,
        operation: str,
        strategy: Callable[[], Any]
    ):
        """Register recovery strategy for operation."""
        self.recovery_strategies[operation] = strategy
        logger.info(f"Registered recovery strategy for {operation}")
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> bool:
        """Coordinate error handling and recovery.
        
        Returns:
            bool: True if error was handled/recovered, False otherwise
        """
        # Update metrics
        self._update_metrics(operation, error, severity)
        
        # Record in history
        self._error_history.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'error': str(error),
            'severity': severity
        })
        
        # Check for cascading failure
        if self._detect_cascade(operation):
            logger.critical(f"Cascade detected in {operation} - triggering prevention")
            self._prevent_cascade(operation)
            return False
        
        # Try specific handler
        handler = self.error_handlers.get(type(error))
        if handler:
            try:
                handler(error)
                return True
            except Exception as handler_error:
                logger.error(f"Handler failed: {handler_error}")
        
        # Try recovery strategy
        if operation in self.recovery_strategies:
            try:
                self.recovery_strategies[operation]()
                self._update_recovery_metrics(operation, success=True)
                return True
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                self._update_recovery_metrics(operation, success=False)
        
        return False
    
    def _update_metrics(
        self,
        operation: str,
        error: Exception,
        severity: ErrorSeverity
    ):
        """Update error metrics for operation."""
        if operation not in self.error_metrics:
            self.error_metrics[operation] = ErrorMetrics()
        
        metrics = self.error_metrics[operation]
        metrics.total_errors += 1
        
        error_type = type(error).__name__
        metrics.errors_by_type[error_type] = \
            metrics.errors_by_type.get(error_type, 0) + 1
        
        # Calculate error rate (errors per minute)
        now = datetime.now()
        if metrics.last_error:
            time_diff = (now - metrics.last_error).total_seconds()
            if time_diff > 0:
                metrics.error_rate = 60.0 / time_diff
        
        metrics.last_error = now
    
    def _detect_cascade(self, operation: str) -> bool:
        """Detect potential cascading failure."""
        # Check error rate
        metrics = self.error_metrics.get(operation)
        if not metrics:
            return False
        
        # Cascade detection thresholds
        if metrics.error_rate > 10:  # More than 10 errors per minute
            return True
        
        if metrics.total_errors > 50 and metrics.successful_recoveries < 5:
            return True
        
        # Check related operations failing
        related_failing = sum(
            1 for op, m in self.error_metrics.items()
            if op != operation and m.error_rate > 5
        )
        
        return related_failing >= 3
    
    def _prevent_cascade(self, operation: str):
        """Prevent cascading failure."""
        # Open circuit breakers for operation
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation].open()
        
        # Notify monitoring
        logger.critical(
            f"Cascade prevention activated for {operation}",
            extra={
                'alert': 'cascade_prevention',
                'operation': operation,
                'metrics': self.error_metrics.get(operation)
            }
        )
    
    def _update_recovery_metrics(self, operation: str, success: bool):
        """Update recovery metrics."""
        if operation not in self.error_metrics:
            self.error_metrics[operation] = ErrorMetrics()
        
        metrics = self.error_metrics[operation]
        metrics.recovery_attempts += 1
        
        if success:
            metrics.successful_recoveries += 1
            # Calculate MTTR
            if metrics.last_error:
                recovery_time = (datetime.now() - metrics.last_error).total_seconds()
                # Exponential moving average
                alpha = 0.3
                metrics.mttr_seconds = (
                    alpha * recovery_time + 
                    (1 - alpha) * metrics.mttr_seconds
                )


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_attempts = 0
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
            else:
                raise RuntimeError(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def call_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute async function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
            else:
                raise RuntimeError(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (
            datetime.now() - self.last_failure_time
        ).total_seconds()
        
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.config.half_open_attempts:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} reopened")
        elif self.failure_count >= self.config.failure_threshold:
            self.open()
    
    def open(self):
        """Open circuit breaker."""
        self.state = CircuitState.OPEN
        logger.warning(
            f"Circuit breaker {self.name} OPENED after "
            f"{self.failure_count} failures"
        )


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_attempts} attempts failed"
                        )
            
            raise last_exception
        return wrapper
    
    def retry_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for async retry logic."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_attempts} attempts failed"
                        )
            
            raise last_exception
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay


# Global coordinator instance
error_coordinator = ErrorCoordinator()

# Pre-configured strategies
default_retry = RetryStrategy()
aggressive_retry = RetryStrategy(
    RetryConfig(max_attempts=5, base_delay=0.5)
)
conservative_retry = RetryStrategy(
    RetryConfig(max_attempts=2, base_delay=5.0)
)


def with_error_coordination(
    operation: str,
    circuit_breaker: bool = True,
    retry: bool = True,
    fallback: Optional[Callable] = None
):
    """Decorator for comprehensive error coordination."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Setup circuit breaker if needed
        if circuit_breaker and operation not in error_coordinator.circuit_breakers:
            error_coordinator.circuit_breakers[operation] = CircuitBreaker(operation)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                # Apply retry strategy
                exec_func = func
                if retry:
                    exec_func = default_retry.retry(func)
                
                # Apply circuit breaker
                if circuit_breaker:
                    cb = error_coordinator.circuit_breakers[operation]
                    return cb.call(exec_func, *args, **kwargs)
                else:
                    return exec_func(*args, **kwargs)
                    
            except Exception as e:
                # Coordinate error handling
                handled = error_coordinator.handle_error(
                    e,
                    operation,
                    ErrorSeverity.HIGH
                )
                
                if not handled and fallback:
                    logger.warning(f"Using fallback for {operation}")
                    return fallback(*args, **kwargs)
                
                raise
        
        return wrapper
    return decorator


# Export main components
__all__ = [
    'ErrorCoordinator',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'RetryStrategy',
    'RetryConfig',
    'ErrorSeverity',
    'ErrorMetrics',
    'error_coordinator',
    'with_error_coordination',
    'default_retry',
    'aggressive_retry',
    'conservative_retry'
]
