"""Legacy configuration bridge for backward compatibility."""

from __future__ import annotations

from typing import Any

from .factory import ConfigFactory, ConfigType


class Config:
    """Compatibility layer mirroring the pre-refactor Config API."""

    @staticmethod
    def create_from_env() -> dict[str, Any]:
        """Return config instances eager-loaded from environment variables."""
        return {key.value: ConfigFactory.get_config(key) for key in ConfigType}

    @staticmethod
    def get_config(config_type: str, **kwargs: Any) -> Any:
        """Return a specific configuration instance by string key."""
        try:
            enum_value = ConfigType(config_type.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown configuration type: {config_type}") from exc
        return ConfigFactory.get_config(enum_value, **kwargs)

    @staticmethod
    def get_provider_config(provider_name: str, **kwargs: Any) -> Any:
        """Compatibility wrapper mapping to provider config lookup."""
        return Config.get_config(provider_name, **kwargs)

    @staticmethod
    def validate(config_type: str | None = None) -> bool:
        """Best-effort validation hook retaining legacy semantics."""
        if config_type is None:
            ConfigFactory.get_all_configs()
            return True
        Config.get_config(config_type)
        return True


# Convenience exports expected by legacy imports
get_config = Config.get_config
