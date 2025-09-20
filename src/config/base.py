"""Base configuration module for environment loading and core settings."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)


# Configuration overlay priorities
class ConfigPriority(Enum):
    """Configuration priority levels for overlay system."""

    DEFAULTS = 0
    FILE = 1
    ENVIRONMENT = 2
    CLI = 3


T = TypeVar("T", bound="BaseConfig")


@dataclass
class ConfigurationSchema:
    """Schema definition for configuration validation."""

    name: str
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, callable] = field(default_factory=dict)

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in config:
                raise ValueError(f"Required field '{field_name}' missing in {self.name}")

        # Apply validators
        for field_name, validator_fn in self.validators.items():
            if field_name in config:
                if not validator_fn(config[field_name]):
                    raise ValueError(f"Validation failed for field '{field_name}' in {self.name}")

        return True


class BaseConfig(ABC):
    """Abstract base class for configuration modules."""

    _instances: Dict[Type, Any] = {}
    _lock = Lock()
    _hot_reload_enabled = False
    _config_overlays: Dict[ConfigPriority, Dict[str, Any]] = {}
    _watchers: Dict[str, callable] = {}

    def __init__(self):
        """Initialize configuration with environment loading."""
        self._load_environment()
        self._cached_values: Dict[str, Any] = {}
        self._schema: Optional[ConfigurationSchema] = None

    @classmethod
    def get_instance(cls: Type[T]) -> T:
        """Get singleton instance of configuration class (thread-safe).

        Returns:
            Singleton instance of the configuration class
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls()
        return cls._instances[cls]

    def _load_environment(self) -> None:
        """Load environment variables and .env file if available."""
        try:
            from dotenv import load_dotenv

            # Search for .env file in current and parent directories
            env_paths = [Path(".env"), Path("../.env"), Path.home() / ".env"]

            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path, override=False)
                    logger.debug(f"Loaded environment from {env_path}")
                    break
        except ImportError:
            logger.debug("python-dotenv not installed, using system environment only")

    @abstractmethod
    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema for validation.

        Returns:
            ConfigurationSchema instance
        """
        pass

    def validate(self) -> bool:
        """Validate configuration against schema.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if self._schema is None:
            self._schema = self.get_schema()

        config_dict = self.to_dict()
        return self._schema.validate(config_dict)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        pass

    @classmethod
    def set_overlay(cls, priority: ConfigPriority, config: Dict[str, Any]) -> None:
        """Set configuration overlay at specified priority.

        Args:
            priority: Priority level for overlay
            config: Configuration dictionary
        """
        cls._config_overlays[priority] = config
        cls._notify_watchers("overlay_changed", {"priority": priority, "config": config})

    @classmethod
    def get_value(cls, key: str, default: Any = None) -> Any:
        """Get configuration value with overlay priority.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        # Check overlays in priority order (highest to lowest)
        for priority in sorted(ConfigPriority, key=lambda p: p.value, reverse=True):
            if priority in cls._config_overlays:
                overlay = cls._config_overlays[priority]
                if key in overlay:
                    return overlay[key]

        # Fall back to environment
        env_value = os.getenv(key.upper())
        if env_value is not None:
            return env_value

        return default

    @classmethod
    def enable_hot_reload(cls, enabled: bool = True) -> None:
        """Enable or disable hot reload of configuration.

        Args:
            enabled: Whether to enable hot reload
        """
        cls._hot_reload_enabled = enabled
        if enabled:
            cls._start_file_watcher()
        else:
            cls._stop_file_watcher()

    @classmethod
    def _start_file_watcher(cls) -> None:
        """Start watching configuration files for changes."""
        # Implementation would use watchdog or similar
        # For now, this is a placeholder
        logger.info("Hot reload enabled for configuration")

    @classmethod
    def _stop_file_watcher(cls) -> None:
        """Stop watching configuration files."""
        logger.info("Hot reload disabled for configuration")

    @classmethod
    def add_watcher(cls, name: str, callback: callable) -> None:
        """Add configuration change watcher.

        Args:
            name: Watcher name
            callback: Callback function to call on changes
        """
        cls._watchers[name] = callback

    @classmethod
    def remove_watcher(cls, name: str) -> None:
        """Remove configuration change watcher.

        Args:
            name: Watcher name to remove
        """
        if name in cls._watchers:
            del cls._watchers[name]

    @classmethod
    def _notify_watchers(cls, event: str, data: Dict[str, Any]) -> None:
        """Notify all watchers of configuration change.

        Args:
            event: Event type
            data: Event data
        """
        for name, callback in cls._watchers.items():
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Error in configuration watcher '{name}': {e}")

    def get_with_fallback(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value with multiple fallback keys.

        Args:
            *keys: Keys to try in order
            default: Default value if none found

        Returns:
            First found value or default
        """
        for key in keys:
            value = self.get_value(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def parse_bool(value: Any) -> bool:
        """Parse boolean value from various formats.

        Args:
            value: Value to parse

        Returns:
            Boolean value
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on", "enabled")
        return bool(value)

    @staticmethod
    def parse_list(value: Any, delimiter: str = ",") -> list:
        """Parse list value from string or return as-is if already a list.

        Args:
            value: Value to parse
            delimiter: String delimiter

        Returns:
            List value
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(delimiter) if item.strip()]
        return [value] if value is not None else []

    @staticmethod
    def sanitize_for_logging(value: str, min_length: int = 8) -> str:
        """Sanitize sensitive values for logging.

        Args:
            value: Value to sanitize
            min_length: Minimum length to show partial value

        Returns:
            Sanitized value safe for logging
        """
        if not value or len(value) < min_length:
            return "[REDACTED]"

        # Show first 4 and last 4 characters
        visible_chars = min(4, len(value) // 4)
        return f"{value[:visible_chars]}{'*' * (len(value) - 2 * visible_chars)}{value[-visible_chars:]}"


class GlobalConfig(BaseConfig):
    """Global application configuration."""

    def __init__(self):
        """Initialize global configuration."""
        super().__init__()

        # Application metadata
        self.app_name = self.get_value("APP_NAME", "audio-extraction-analysis")
        self.app_version = self.get_value("APP_VERSION", "1.0.0")
        self.environment = self.get_value("ENVIRONMENT", "production")

        # Paths
        self.data_dir = Path(self.get_value("DATA_DIR", "./data"))
        self.cache_dir = Path(self.get_value("CACHE_DIR", "./cache"))
        self.temp_dir = Path(self.get_value("TEMP_DIR", "/tmp"))

        # File handling
        self.max_file_size = int(self.get_value("MAX_FILE_SIZE", "100000000"))  # 100MB
        self.allowed_extensions = self.parse_list(
            self.get_value("ALLOWED_EXTENSIONS", ".mp3,.wav,.m4a,.flac,.ogg,.aac")
        )

        # Logging
        self.log_level = self.get_value("LOG_LEVEL", "INFO").upper()
        self.log_format = self.get_value(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_file = self.get_value("LOG_FILE")

        # Provider selection
        self.default_provider = self.get_value("DEFAULT_TRANSCRIPTION_PROVIDER", "deepgram")
        self.fallback_providers = self.parse_list(
            self.get_value("FALLBACK_PROVIDERS", "elevenlabs,whisper")
        )

        # Language settings
        self.default_language = self.get_value("DEFAULT_LANGUAGE", "en")
        self.supported_languages = self.parse_list(
            self.get_value("SUPPORTED_LANGUAGES", "en,es,fr,de,it,pt,ru,ja,ko,zh")
        )

        # Feature flags
        self.enable_caching = self.parse_bool(self.get_value("ENABLE_CACHING", "true"))
        self.enable_retries = self.parse_bool(self.get_value("ENABLE_RETRIES", "true"))
        self.enable_health_checks = self.parse_bool(self.get_value("ENABLE_HEALTH_CHECKS", "true"))
        self.enable_metrics = self.parse_bool(self.get_value("ENABLE_METRICS", "false"))

        # Create directories if they don't exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for directory in [self.data_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for global config
        """
        return ConfigurationSchema(
            name="GlobalConfig",
            required_fields={"app_name", "environment"},
            optional_fields={
                "app_version": "1.0.0",
                "max_file_size": 100000000,
                "log_level": "INFO",
            },
            validators={
                "log_level": lambda x: x in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
                "environment": lambda x: x in {"development", "staging", "production", "test"},
                "max_file_size": lambda x: isinstance(x, int) and x > 0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "temp_dir": str(self.temp_dir),
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.allowed_extensions,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file,
            "default_provider": self.default_provider,
            "fallback_providers": self.fallback_providers,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "enable_caching": self.enable_caching,
            "enable_retries": self.enable_retries,
            "enable_health_checks": self.enable_health_checks,
            "enable_metrics": self.enable_metrics,
        }

    def validate_file_path(self, file_path: Path) -> bool:
        """Validate file path against security settings.

        Args:
            file_path: Path to validate

        Returns:
            True if valid, False otherwise
        """
        # Check extension
        if file_path.suffix.lower() not in self.allowed_extensions:
            logger.warning(f"Invalid file extension: {file_path.suffix}")
            return False

        # Check file size if it exists
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_size} > {self.max_file_size}")
                return False

        return True


# Singleton instance getter
def get_global_config() -> GlobalConfig:
    """Get global configuration instance.

    Returns:
        GlobalConfig singleton instance
    """
    return GlobalConfig.get_instance()
