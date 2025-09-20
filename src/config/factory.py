"""Unified configuration factory to eliminate duplicate config getters."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from .base import BaseConfig

T = TypeVar('T', bound=BaseConfig)


class ConfigType(Enum):
    """Available configuration types."""
    GLOBAL = "global"
    UI = "ui"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ELEVENLABS = "elevenlabs"
    PARAKEET = "parakeet"
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"


class ConfigFactory:
    """Factory for creating and managing configuration instances."""
    
    _instances: Dict[ConfigType, BaseConfig] = {}
    _config_classes: Dict[ConfigType, Type[BaseConfig]] = {}
    
    @classmethod
    def register(cls, config_type: ConfigType, config_class: Type[BaseConfig]) -> None:
        """Register a configuration class for a given type.
        
        Args:
            config_type: The configuration type
            config_class: The configuration class to register
        """
        cls._config_classes[config_type] = config_class
    
    @classmethod
    def get_config(cls, config_type: ConfigType, **kwargs) -> BaseConfig:
        """Get or create a configuration instance.
        
        Args:
            config_type: The type of configuration to get
            **kwargs: Additional arguments to pass to the configuration constructor
            
        Returns:
            The configuration instance
            
        Raises:
            ValueError: If configuration type is not registered
        """
        # Return existing instance if available (singleton pattern)
        if config_type in cls._instances:
            return cls._instances[config_type]
        
        # Create new instance
        if config_type not in cls._config_classes:
            # Lazy import to avoid circular dependencies
            cls._lazy_register_configs()
            
            if config_type not in cls._config_classes:
                raise ValueError(f"Configuration type {config_type.value} not registered")
        
        config_class = cls._config_classes[config_type]
        instance = config_class(**kwargs)
        cls._instances[config_type] = instance
        
        return instance
    
    @classmethod
    def _lazy_register_configs(cls) -> None:
        """Lazy registration of configuration classes to avoid circular imports."""
        # Import and register configuration classes
        try:
            from .base import get_global_config
            from .ui import UIConfig
            from .security import SecurityConfig
            from .performance import PerformanceConfig
            from .providers.elevenlabs import ElevenLabsConfig
            from .providers.parakeet import ParakeetConfig
            from .providers.deepgram import DeepgramConfig
            from .providers.whisper import WhisperConfig
            
            # Register each config type
            # Note: Using the actual config classes, not the get_* functions
            # The classes need to be properly imported
            
            # For now, we'll create a mapping that can be extended
            config_map = {
                ConfigType.UI: UIConfig,
                ConfigType.SECURITY: SecurityConfig,
                ConfigType.PERFORMANCE: PerformanceConfig,
                ConfigType.ELEVENLABS: ElevenLabsConfig,
                ConfigType.PARAKEET: ParakeetConfig,
                ConfigType.DEEPGRAM: DeepgramConfig,
                ConfigType.WHISPER: WhisperConfig,
            }
            
            for config_type, config_class in config_map.items():
                if config_type not in cls._config_classes:
                    cls.register(config_type, config_class)
                    
        except ImportError as e:
            # Log but don't fail - allow partial registration
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some config types: {e}")
    
    @classmethod
    def reset(cls) -> None:
        """Reset all cached configuration instances (useful for testing)."""
        cls._instances.clear()
    
    @classmethod
    def get_all_configs(cls) -> Dict[ConfigType, BaseConfig]:
        """Get all registered configuration instances.
        
        Returns:
            Dictionary of all configuration instances
        """
        cls._lazy_register_configs()
        result = {}
        for config_type in cls._config_classes:
            try:
                result[config_type] = cls.get_config(config_type)
            except Exception:
                pass  # Skip configs that fail to initialize
        return result


# Convenience functions for backward compatibility
def get_config(config_type: str, **kwargs) -> BaseConfig:
    """Get configuration by string type name.
    
    Args:
        config_type: Configuration type name (e.g., "ui", "security")
        **kwargs: Additional arguments
        
    Returns:
        Configuration instance
    """
    try:
        config_enum = ConfigType(config_type.lower())
        return ConfigFactory.get_config(config_enum, **kwargs)
    except ValueError:
        raise ValueError(f"Unknown configuration type: {config_type}")


def get_provider_config(provider_name: str, **kwargs) -> BaseConfig:
    """Get provider-specific configuration.
    
    Args:
        provider_name: Provider name (e.g., "deepgram", "elevenlabs")
        **kwargs: Additional arguments
        
    Returns:
        Provider configuration instance
    """
    return get_config(provider_name, **kwargs)