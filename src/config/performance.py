"""Performance-related configuration settings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional

from .base import BaseConfig, ConfigurationSchema

logger = logging.getLogger(__name__)


class ConcurrencyMode(Enum):
    """Concurrency execution modes."""

    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    ASYNC = "async"
    PROCESS = "process"


@dataclass
class PerformanceProfile:
    """Performance profile for different scenarios."""

    name: str
    max_workers: int
    timeout: int
    batch_size: int
    memory_limit: Optional[int]  # MB
    cpu_limit: Optional[float]  # 0.0 to 1.0
    priority: int  # 0 (lowest) to 10 (highest)


class PerformanceConfig(BaseConfig):
    """Performance and optimization settings."""

    # Predefined performance profiles
    _PROFILES = {
        "low": PerformanceProfile(
            name="low",
            max_workers=1,
            timeout=300,
            batch_size=1,
            memory_limit=512,
            cpu_limit=0.25,
            priority=1,
        ),
        "medium": PerformanceProfile(
            name="medium",
            max_workers=4,
            timeout=600,
            batch_size=5,
            memory_limit=2048,
            cpu_limit=0.5,
            priority=5,
        ),
        "high": PerformanceProfile(
            name="high",
            max_workers=8,
            timeout=1200,
            batch_size=10,
            memory_limit=4096,
            cpu_limit=0.75,
            priority=7,
        ),
        "maximum": PerformanceProfile(
            name="maximum",
            max_workers=16,
            timeout=3600,
            batch_size=20,
            memory_limit=None,
            cpu_limit=1.0,
            priority=10,
        ),
    }

    def __init__(self):
        """Initialize performance configuration."""
        super().__init__()

        # Performance profile
        profile_name = self.get_value("PERFORMANCE_PROFILE", "medium")
        self.profile = self._load_profile(profile_name)

        # Concurrency settings
        self.concurrency_mode = ConcurrencyMode(self.get_value("CONCURRENCY_MODE", "async").lower())
        self.max_workers = int(self.get_value("MAX_WORKERS", str(self.profile.max_workers)))
        self.max_concurrent_requests = int(self.get_value("MAX_CONCURRENT_REQUESTS", "10"))
        self.thread_pool_size = int(self.get_value("THREAD_POOL_SIZE", "10"))
        self.process_pool_size = int(self.get_value("PROCESS_POOL_SIZE", "4"))

        # Timeout settings
        self.global_timeout = int(self.get_value("GLOBAL_TIMEOUT", str(self.profile.timeout)))
        self.connect_timeout = int(self.get_value("CONNECT_TIMEOUT", "10"))
        self.read_timeout = int(self.get_value("READ_TIMEOUT", "30"))
        self.write_timeout = int(self.get_value("WRITE_TIMEOUT", "30"))

        # Retry settings
        self.max_retries = int(self.get_value("MAX_API_RETRIES", "3"))
        self.retry_delay = float(self.get_value("API_RETRY_DELAY", "1.0"))
        self.max_retry_delay = float(self.get_value("MAX_RETRY_DELAY", "60.0"))
        self.retry_exponential_base = float(self.get_value("RETRY_EXPONENTIAL_BASE", "2.0"))
        self.retry_jitter = self.parse_bool(self.get_value("RETRY_JITTER_ENABLED", "true"))

        # Circuit breaker settings
        self.circuit_breaker_enabled = self.parse_bool(
            self.get_value("CIRCUIT_BREAKER_ENABLED", "true")
        )
        self.circuit_breaker_failure_threshold = int(
            self.get_value("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
        )
        self.circuit_breaker_recovery_timeout = float(
            self.get_value("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0")
        )
        self.circuit_breaker_half_open_requests = int(
            self.get_value("CIRCUIT_BREAKER_HALF_OPEN_REQUESTS", "3")
        )

        # Batch processing
        self.batch_size = int(self.get_value("BATCH_SIZE", str(self.profile.batch_size)))
        self.batch_timeout = int(self.get_value("BATCH_TIMEOUT", "60"))
        self.batch_max_wait = float(self.get_value("BATCH_MAX_WAIT", "5.0"))

        # Memory management
        self.memory_limit_mb = self._parse_memory_limit(
            self.get_value("MEMORY_LIMIT", str(self.profile.memory_limit or "0"))
        )
        self.gc_threshold = int(self.get_value("GC_THRESHOLD", "100"))
        self.enable_memory_profiling = self.parse_bool(
            self.get_value("ENABLE_MEMORY_PROFILING", "false")
        )

        # CPU management
        self.cpu_limit = self._parse_cpu_limit(
            self.get_value("CPU_LIMIT", str(self.profile.cpu_limit or "1.0"))
        )
        self.cpu_affinity = self.parse_list(self.get_value("CPU_AFFINITY", ""))

        # Connection pooling
        self.connection_pool_size = int(self.get_value("CONNECTION_POOL_SIZE", "10"))
        self.connection_max_age = int(self.get_value("CONNECTION_MAX_AGE", "300"))
        self.connection_timeout = int(self.get_value("CONNECTION_TIMEOUT", "10"))

        # Caching
        self.cache_ttl = int(self.get_value("CACHE_TTL", "3600"))
        self.cache_max_size = int(self.get_value("CACHE_MAX_SIZE", "1000"))
        self.cache_eviction_policy = self.get_value("CACHE_EVICTION_POLICY", "lru")

        # Request queuing
        self.queue_size = int(self.get_value("QUEUE_SIZE", "100"))
        self.queue_timeout = int(self.get_value("QUEUE_TIMEOUT", "30"))
        self.queue_priority_enabled = self.parse_bool(
            self.get_value("QUEUE_PRIORITY_ENABLED", "false")
        )

        # Performance monitoring
        self.enable_metrics = self.parse_bool(self.get_value("ENABLE_METRICS", "false"))
        self.metrics_interval = int(self.get_value("METRICS_INTERVAL", "60"))
        self.enable_tracing = self.parse_bool(self.get_value("ENABLE_TRACING", "false"))
        self.trace_sample_rate = float(self.get_value("TRACE_SAMPLE_RATE", "0.1"))

        # Resource limits tracking
        self._resource_usage = {"cpu": 0.0, "memory": 0.0, "connections": 0, "threads": 0}
        self._resource_lock = Lock()

    def _load_profile(self, profile_name: str) -> PerformanceProfile:
        """Load performance profile by name.

        Args:
            profile_name: Profile name or "custom"

        Returns:
            PerformanceProfile instance
        """
        if profile_name in self._PROFILES:
            return self._PROFILES[profile_name]

        # Custom profile from environment
        return PerformanceProfile(
            name="custom",
            max_workers=int(self.get_value("CUSTOM_MAX_WORKERS", "4")),
            timeout=int(self.get_value("CUSTOM_TIMEOUT", "600")),
            batch_size=int(self.get_value("CUSTOM_BATCH_SIZE", "5")),
            memory_limit=self._parse_memory_limit(self.get_value("CUSTOM_MEMORY_LIMIT", "0")),
            cpu_limit=self._parse_cpu_limit(self.get_value("CUSTOM_CPU_LIMIT", "1.0")),
            priority=int(self.get_value("CUSTOM_PRIORITY", "5")),
        )

    def _parse_memory_limit(self, value: str) -> Optional[int]:
        """Parse memory limit string to MB.

        Args:
            value: Memory limit string (e.g., "2GB", "512MB", "0")

        Returns:
            Memory limit in MB or None for unlimited
        """
        if not value or value == "0":
            return None

        value = value.upper()
        if value.endswith("GB"):
            return int(float(value[:-2]) * 1024)
        elif value.endswith("MB"):
            return int(value[:-2])
        elif value.endswith("KB"):
            return int(float(value[:-2]) / 1024)
        else:
            return int(value)

    def _parse_cpu_limit(self, value: str) -> float:
        """Parse CPU limit string to float.

        Args:
            value: CPU limit string (percentage or decimal)

        Returns:
            CPU limit as float (0.0 to 1.0)
        """
        if value.endswith("%"):
            return float(value[:-1]) / 100
        else:
            return max(0.0, min(1.0, float(value)))

    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration dictionary.

        Returns:
            Retry configuration
        """
        return {
            "max_attempts": self.max_retries,
            "initial_delay": self.retry_delay,
            "max_delay": self.max_retry_delay,
            "exponential_base": self.retry_exponential_base,
            "jitter": self.retry_jitter,
        }

    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration dictionary.

        Returns:
            Circuit breaker configuration
        """
        return {
            "enabled": self.circuit_breaker_enabled,
            "failure_threshold": self.circuit_breaker_failure_threshold,
            "recovery_timeout": self.circuit_breaker_recovery_timeout,
            "half_open_requests": self.circuit_breaker_half_open_requests,
        }

    def get_timeout_config(self) -> Dict[str, Any]:
        """Get timeout configuration dictionary.

        Returns:
            Timeout configuration
        """
        return {
            "global": self.global_timeout,
            "connect": self.connect_timeout,
            "read": self.read_timeout,
            "write": self.write_timeout,
            "total": self.global_timeout,
        }

    def get_concurrency_config(self) -> Dict[str, Any]:
        """Get concurrency configuration dictionary.

        Returns:
            Concurrency configuration
        """
        return {
            "mode": self.concurrency_mode.value,
            "max_workers": self.max_workers,
            "max_concurrent_requests": self.max_concurrent_requests,
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
        }

    def update_resource_usage(self, resource: str, value: float) -> None:
        """Update resource usage tracking.

        Args:
            resource: Resource type (cpu, memory, connections, threads)
            value: Current usage value
        """
        with self._resource_lock:
            self._resource_usage[resource] = value

            # Check limits
            if resource == "cpu" and self.cpu_limit:
                if value > self.cpu_limit:
                    logger.warning(f"CPU usage {value:.2%} exceeds limit {self.cpu_limit:.2%}")

            elif resource == "memory" and self.memory_limit_mb:
                if value > self.memory_limit_mb:
                    logger.warning(f"Memory usage {value}MB exceeds limit {self.memory_limit_mb}MB")

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage.

        Returns:
            Dictionary of resource usage metrics
        """
        with self._resource_lock:
            return self._resource_usage.copy()

    def should_throttle(self) -> bool:
        """Check if processing should be throttled based on resource usage.

        Returns:
            True if throttling is needed
        """
        with self._resource_lock:
            # Check CPU limit
            if self.cpu_limit and self._resource_usage["cpu"] > self.cpu_limit * 0.9:
                return True

            # Check memory limit
            if self.memory_limit_mb and self._resource_usage["memory"] > self.memory_limit_mb * 0.9:
                return True

            return False

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current resource usage.

        Returns:
            Recommended batch size
        """
        if self.should_throttle():
            # Reduce batch size when throttling
            return max(1, self.batch_size // 2)

        return self.batch_size

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for performance config
        """
        return ConfigurationSchema(
            name="PerformanceConfig",
            required_fields=set(),
            optional_fields={
                "max_workers": 4,
                "global_timeout": 600,
                "max_retries": 3,
                "batch_size": 5,
                "cache_ttl": 3600,
            },
            validators={
                "max_workers": lambda x: isinstance(x, int) and 1 <= x <= 100,
                "global_timeout": lambda x: isinstance(x, int) and x > 0,
                "max_retries": lambda x: isinstance(x, int) and x >= 0,
                "retry_delay": lambda x: isinstance(x, (int, float)) and x > 0,
                "batch_size": lambda x: isinstance(x, int) and x > 0,
                "cpu_limit": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
                "cache_ttl": lambda x: isinstance(x, int) and x >= 0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "profile": self.profile.name,
            "concurrency_mode": self.concurrency_mode.value,
            "max_workers": self.max_workers,
            "max_concurrent_requests": self.max_concurrent_requests,
            "global_timeout": self.global_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "batch_size": self.batch_size,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit": self.cpu_limit,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
        }


# Singleton instance getter
def get_performance_config() -> PerformanceConfig:
    """Get performance configuration instance.

    Returns:
        PerformanceConfig singleton instance
    """
    return PerformanceConfig.get_instance()
