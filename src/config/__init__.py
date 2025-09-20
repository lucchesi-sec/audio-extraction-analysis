"""Configuration management facade for backward compatibility.

This module provides a backward-compatible interface to the refactored configuration system.
Existing code can continue using the Config class while benefiting from the modular architecture.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import modular configuration components
from .base import ConfigPriority, GlobalConfig, get_global_config
from .performance import PerformanceConfig, get_performance_config

# Import provider configurations
from .providers import (
    DeepgramConfig,
    ElevenLabsConfig,
    ParakeetConfig,
    WhisperConfig,
    get_deepgram_config,
    get_elevenlabs_config,
    get_parakeet_config,
    get_whisper_config,
)
from .security import SecurityConfig, get_security_config
from .ui import UIConfig, get_ui_config
from .validation import (
    CircuitBreakerConfigModel,
    ConfigValidator,
    FileConfigModel,
    RetryConfigModel,
    ValidationLevel,
    create_config_validator,
)

logger = logging.getLogger(__name__)


class Config:
    """Backward-compatible configuration class that delegates to modular components.

    This class maintains the original interface while using the new modular
    configuration system under the hood.
    """

    # Class-level instances for singleton pattern
    _global_config: Optional[GlobalConfig] = None
    _security_config: Optional[SecurityConfig] = None
    _performance_config: Optional[PerformanceConfig] = None
    _ui_config: Optional[UIConfig] = None
    _provider_configs: Dict[str, Any] = {}
    _validator: Optional[ConfigValidator] = None

    def __init__(self):
        """Initialize configuration by loading all modules."""
        # Initialize modular configurations
        self._global_config = get_global_config()
        self._security_config = get_security_config()
        self._performance_config = get_performance_config()
        self._ui_config = get_ui_config()

        # Initialize validator
        validation_level = os.getenv("CONFIG_VALIDATION_LEVEL", "normal")
        self._validator = create_config_validator(validation_level)

        # Cache provider configurations lazily
        self._provider_configs = {}

        # Log configuration loading
        logger.info("Configuration system initialized with modular architecture")

    # ========== API Key Properties (Backward Compatibility) ==========

    @property
    def DEEPGRAM_API_KEY(self) -> Optional[str]:
        """Get Deepgram API key."""
        try:
            return self._security_config.get_api_key("deepgram", validate=False)
        except ValueError:
            return None

    @property
    def ELEVENLABS_API_KEY(self) -> Optional[str]:
        """Get ElevenLabs API key."""
        try:
            return self._security_config.get_api_key("elevenlabs", validate=False)
        except ValueError:
            return None

    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        """Get Gemini API key."""
        try:
            return self._security_config.get_api_key("gemini", validate=False)
        except ValueError:
            return None

    # ========== Provider Configuration Properties ==========

    @property
    def WHISPER_MODEL(self) -> str:
        """Get Whisper model."""
        if "whisper" not in self._provider_configs:
            self._provider_configs["whisper"] = get_whisper_config()
        return self._provider_configs["whisper"].model.value

    @property
    def WHISPER_DEVICE(self) -> str:
        """Get Whisper device."""
        if "whisper" not in self._provider_configs:
            self._provider_configs["whisper"] = get_whisper_config()
        return self._provider_configs["whisper"].device.value

    @property
    def WHISPER_COMPUTE_TYPE(self) -> str:
        """Get Whisper compute type."""
        if "whisper" not in self._provider_configs:
            self._provider_configs["whisper"] = get_whisper_config()
        return self._provider_configs["whisper"].compute_type.value

    @property
    def PARAKEET_MODEL(self) -> str:
        """Get Parakeet model."""
        if "parakeet" not in self._provider_configs:
            self._provider_configs["parakeet"] = get_parakeet_config()
        return self._provider_configs["parakeet"].model.value

    @property
    def PARAKEET_DEVICE(self) -> str:
        """Get Parakeet device."""
        if "parakeet" not in self._provider_configs:
            self._provider_configs["parakeet"] = get_parakeet_config()
        return self._provider_configs["parakeet"].device

    @property
    def PARAKEET_BATCH_SIZE(self) -> int:
        """Get Parakeet batch size."""
        if "parakeet" not in self._provider_configs:
            self._provider_configs["parakeet"] = get_parakeet_config()
        return self._provider_configs["parakeet"].batch_size

    @property
    def PARAKEET_USE_FP16(self) -> bool:
        """Get Parakeet FP16 setting."""
        if "parakeet" not in self._provider_configs:
            self._provider_configs["parakeet"] = get_parakeet_config()
        return self._provider_configs["parakeet"].use_fp16

    # ========== Global Configuration Properties ==========

    @property
    def DEFAULT_TRANSCRIPTION_PROVIDER(self) -> str:
        """Get default transcription provider."""
        return self._global_config.default_provider

    @property
    def AVAILABLE_PROVIDERS(self) -> List[str]:
        """Get available providers."""
        return ["deepgram", "elevenlabs", "whisper", "parakeet", "auto"]

    @property
    def DEFAULT_LANGUAGE(self) -> str:
        """Get default language."""
        return self._global_config.default_language

    @property
    def MAX_FILE_SIZE(self) -> int:
        """Get maximum file size."""
        return self._global_config.max_file_size

    @property
    def ALLOWED_FILE_EXTENSIONS(self) -> set:
        """Get allowed file extensions."""
        return set(self._global_config.allowed_extensions)

    # ========== Performance Configuration Properties ==========

    @property
    def MAX_API_RETRIES(self) -> int:
        """Get maximum API retries."""
        return self._performance_config.max_retries

    @property
    def API_RETRY_DELAY(self) -> float:
        """Get API retry delay."""
        return self._performance_config.retry_delay

    @property
    def MAX_RETRY_DELAY(self) -> float:
        """Get maximum retry delay."""
        return self._performance_config.max_retry_delay

    @property
    def RETRY_EXPONENTIAL_BASE(self) -> float:
        """Get retry exponential base."""
        return self._performance_config.retry_exponential_base

    @property
    def RETRY_JITTER_ENABLED(self) -> bool:
        """Get retry jitter setting."""
        return self._performance_config.retry_jitter

    @property
    def CIRCUIT_BREAKER_FAILURE_THRESHOLD(self) -> int:
        """Get circuit breaker failure threshold."""
        return self._performance_config.circuit_breaker_failure_threshold

    @property
    def CIRCUIT_BREAKER_RECOVERY_TIMEOUT(self) -> float:
        """Get circuit breaker recovery timeout."""
        return self._performance_config.circuit_breaker_recovery_timeout

    @property
    def HEALTH_CHECK_TIMEOUT(self) -> float:
        """Get health check timeout."""
        return self._performance_config.connect_timeout

    @property
    def HEALTH_CHECK_ENABLED(self) -> bool:
        """Get health check enabled setting."""
        return self._global_config.enable_health_checks

    # ========== Timeout Properties ==========

    @property
    def DEEPGRAM_TIMEOUT(self) -> int:
        """Get Deepgram timeout."""
        if "deepgram" not in self._provider_configs:
            self._provider_configs["deepgram"] = get_deepgram_config()
        return self._provider_configs["deepgram"].timeout

    @property
    def ELEVENLABS_TIMEOUT(self) -> int:
        """Get ElevenLabs timeout."""
        if "elevenlabs" not in self._provider_configs:
            self._provider_configs["elevenlabs"] = get_elevenlabs_config()
        return self._provider_configs["elevenlabs"].timeout

    @property
    def WHISPER_TIMEOUT(self) -> int:
        """Get Whisper timeout."""
        if "whisper" not in self._provider_configs:
            self._provider_configs["whisper"] = get_whisper_config()
        return self._provider_configs["whisper"].timeout

    # ========== Logging Properties ==========

    @property
    def LOG_LEVEL(self) -> str:
        """Get log level."""
        return self._global_config.log_level

    # ========== UI Configuration Properties ==========

    @property
    def markdown_include_timestamps(self) -> bool:
        """Get markdown timestamps setting."""
        return self._ui_config.markdown_include_timestamps

    @property
    def markdown_include_speakers(self) -> bool:
        """Get markdown speakers setting."""
        return self._ui_config.markdown_include_speakers

    @property
    def markdown_include_confidence(self) -> bool:
        """Get markdown confidence setting."""
        return self._ui_config.markdown_include_confidence

    @property
    def markdown_default_template(self) -> str:
        """Get markdown template."""
        return self._ui_config.markdown_template

    # ========== Backward Compatible Methods ==========

    @classmethod
    def validate(cls, provider: str = "deepgram") -> None:
        """Validate configuration for specified provider.

        Args:
            provider: Provider to validate

        Raises:
            ValueError: If configuration is invalid
        """
        instance = cls()

        # Validate global configuration
        instance._global_config.validate()

        # Validate provider-specific configuration
        if provider == "deepgram":
            config = get_deepgram_config()
            if not config.api_key:
                raise ValueError(
                    "DEEPGRAM_API_KEY environment variable not found or invalid. "
                    "Set it in your environment or create a .env file with: "
                    "DEEPGRAM_API_KEY=your-api-key-here"
                )
        elif provider == "elevenlabs":
            config = get_elevenlabs_config()
            if not config.api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY environment variable not found or invalid. "
                    "Set it in your environment or create a .env file with: "
                    "ELEVENLABS_API_KEY=your-api-key-here"
                )
        elif provider == "whisper":
            # Check Whisper dependencies
            try:
                import torch
                import whisper
            except ImportError:
                raise ValueError(
                    "Whisper dependencies not installed. Install with: "
                    "pip install openai-whisper torch"
                )
        elif provider == "parakeet":
            # Check Parakeet/NeMo dependencies
            try:
                import nemo
                import torch
            except ImportError:
                raise ValueError(
                    "Parakeet dependencies not installed. Install with: "
                    "pip install nemo_toolkit torch"
                )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def get_deepgram_api_key(cls) -> str:
        """Get Deepgram API key with validation.

        Returns:
            API key

        Raises:
            ValueError: If not configured
        """
        instance = cls()
        return instance._security_config.get_api_key("deepgram")

    @classmethod
    def get_elevenlabs_api_key(cls) -> str:
        """Get ElevenLabs API key with validation.

        Returns:
            API key

        Raises:
            ValueError: If not configured
        """
        instance = cls()
        return instance._security_config.get_api_key("elevenlabs")

    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key with validation.

        Returns:
            API key

        Raises:
            ValueError: If not configured
        """
        instance = cls()
        return instance._security_config.get_api_key("gemini")

    @classmethod
    def is_configured(cls, provider: Optional[str] = None) -> bool:
        """Check if provider is configured.

        Args:
            provider: Provider to check, or None for any

        Returns:
            True if configured
        """
        instance = cls()

        if provider == "deepgram":
            return instance.DEEPGRAM_API_KEY is not None
        elif provider == "elevenlabs":
            return instance.ELEVENLABS_API_KEY is not None
        elif provider == "whisper":
            try:
                import torch
                import whisper

                return True
            except ImportError:
                return False
        elif provider == "parakeet":
            try:
                import nemo
                import torch

                return True
            except ImportError:
                return False
        else:
            # Check if any provider is configured
            return instance.DEEPGRAM_API_KEY is not None or instance.ELEVENLABS_API_KEY is not None

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of configured providers.

        Returns:
            List of available provider names
        """
        instance = cls()
        available = []

        if instance.DEEPGRAM_API_KEY:
            available.append("deepgram")
        if instance.ELEVENLABS_API_KEY:
            available.append("elevenlabs")

        try:
            import torch
            import whisper

            available.append("whisper")
        except ImportError:
            pass

        try:
            import nemo
            import torch

            available.append("parakeet")
        except ImportError:
            pass

        return available

    @classmethod
    def get_provider_config(cls, provider_name: str) -> Dict[str, Any]:
        """Get provider configuration dictionary.

        Args:
            provider_name: Provider name

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If provider unknown
        """
        instance = cls()

        base_config = {
            "max_retries": instance.MAX_API_RETRIES,
            "retry_delay": instance.API_RETRY_DELAY,
            "max_retry_delay": instance.MAX_RETRY_DELAY,
            "retry_exponential_base": instance.RETRY_EXPONENTIAL_BASE,
            "retry_jitter_enabled": instance.RETRY_JITTER_ENABLED,
            "circuit_breaker_failure_threshold": instance.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "circuit_breaker_recovery_timeout": instance.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            "health_check_timeout": instance.HEALTH_CHECK_TIMEOUT,
            "health_check_enabled": instance.HEALTH_CHECK_ENABLED,
        }

        if provider_name == "deepgram":
            config = get_deepgram_config()
            return {
                **base_config,
                "api_key": config.api_key,
                "timeout": config.timeout,
                "max_file_size": config.max_file_size,
            }
        elif provider_name == "elevenlabs":
            config = get_elevenlabs_config()
            return {
                **base_config,
                "api_key": config.api_key,
                "timeout": config.timeout,
                "max_file_size": config.max_file_size,
            }
        elif provider_name == "whisper":
            config = get_whisper_config()
            return {
                **base_config,
                "api_key": None,
                "timeout": config.timeout,
                "max_file_size": config.max_file_size,
            }
        elif provider_name == "parakeet":
            config = get_parakeet_config()
            return {
                **base_config,
                "api_key": None,
                "timeout": config.timeout,
                "max_file_size": config.max_file_size,
            }
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    @classmethod
    def validate_file_extension(cls, file_path: Path) -> bool:
        """Validate file extension.

        Args:
            file_path: Path to validate

        Returns:
            True if valid
        """
        instance = cls()
        return instance._global_config.validate_file_path(file_path)

    @classmethod
    def get_security_settings(cls) -> Dict[str, Any]:
        """Get security settings.

        Returns:
            Security settings dictionary
        """
        instance = cls()
        return {
            "max_file_size": instance.MAX_FILE_SIZE,
            "allowed_extensions": instance.ALLOWED_FILE_EXTENSIONS,
            "max_retries": instance.MAX_API_RETRIES,
            "timeouts": {
                "deepgram": instance.DEEPGRAM_TIMEOUT,
                "elevenlabs": instance.ELEVENLABS_TIMEOUT,
                "whisper": instance.WHISPER_TIMEOUT,
            },
        }

    @classmethod
    def _sanitize_for_logging(cls, value: str) -> str:
        """Sanitize sensitive values for logging.

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value
        """
        return SecurityConfig.sanitize_for_logging(value)

    @classmethod
    def _validate_deepgram_key(cls, key: str) -> bool:
        """Validate Deepgram key format.

        Args:
            key: API key

        Returns:
            True if valid
        """
        instance = cls()
        return instance._security_config.validate_api_key("deepgram", key)

    @classmethod
    def _validate_elevenlabs_key(cls, key: str) -> bool:
        """Validate ElevenLabs key format.

        Args:
            key: API key

        Returns:
            True if valid
        """
        instance = cls()
        return instance._security_config.validate_api_key("elevenlabs", key)

    @classmethod
    def _validate_gemini_key(cls, key: str) -> bool:
        """Validate Gemini key format.

        Args:
            key: API key

        Returns:
            True if valid
        """
        instance = cls()
        return instance._security_config.validate_api_key("gemini", key)

    @classmethod
    def _validate_configuration(cls) -> None:
        """Validate overall configuration.

        Raises:
            ValueError: If invalid
        """
        instance = cls()
        instance._global_config.validate()
        instance._performance_config.validate()


# UI configuration helper (backward compatibility)
DEFAULT_CONFIG: Dict[str, Any] = {
    "ui": {
        "progress_bars": True,
        "rich_output": True,
        "json_fallback": False,
    }
}

def get_ui_config_dict() -> Dict[str, Any]:
    """Get UI configuration as a simple dictionary.

    This preserves backward compatibility for legacy callers that expected
    a dict while avoiding name collisions with the UI singleton accessor.
    """
    ui = get_ui_config()  # imported from .ui at module top
    return {
        "verbose": ui.verbose,
        "json_output": ui.output_format.value == "json",
        "no_color": ui.no_color,
    }


# Export main Config class and helpers for backward compatibility
__all__ = [
    "DEFAULT_CONFIG",
    "Config",
    "ConfigPriority",
    "ConfigValidator",
    "DeepgramConfig",
    "ElevenLabsConfig",
    # Export modular components for advanced usage
    "GlobalConfig",
    "ParakeetConfig",
    "PerformanceConfig",
    "SecurityConfig",
    "UIConfig",
    "ValidationLevel",
    "WhisperConfig",
    "create_config_validator",
    "get_deepgram_config",
    "get_elevenlabs_config",
    "get_global_config",
    "get_parakeet_config",
    "get_performance_config",
    "get_security_config",
    "get_ui_config",
    "get_whisper_config",
]
