"""Secure configuration management with validation.

This module provides a secure configuration system using Pydantic Settings with:
- API key masking via SecretStr
- Comprehensive validation with clear error messages
- Backward compatibility with .env files
- Support for environment variable loading
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, SecretStr, field_validator


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ProviderConfig(BaseSettings):
    """Provider-specific configuration with secure API key management.

    API keys are wrapped in SecretStr to prevent accidental logging or serialization.
    Access keys via .get_secret_value() method when needed.
    """

    # API Keys (never logged, masked in errors)
    deepgram_api_key: Optional[SecretStr] = Field(None, env='DEEPGRAM_API_KEY')
    elevenlabs_api_key: Optional[SecretStr] = Field(None, env='ELEVENLABS_API_KEY')

    # Provider settings
    whisper_model: Literal['tiny', 'base', 'small', 'medium', 'large'] = 'base'
    whisper_device: Literal['cuda', 'cpu'] = 'cpu'

    # Transcription settings
    default_language: str = 'en'

    @field_validator('deepgram_api_key', 'elevenlabs_api_key')
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate API key format if provided."""
        if v is not None:
            secret_value = v.get_secret_value()
            if len(secret_value) < 10:
                raise ValueError(
                    "API key appears invalid (too short). "
                    "Expected format: provider-specific key with minimum 10 characters. "
                    "Check your .env file or environment variables."
                )
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # Prevent accidental logging of secrets
        json_encoders = {
            SecretStr: lambda v: '***' if v else None
        }


class SecurityConfig(BaseSettings):
    """Security-related configuration."""

    # Temporary file settings
    temp_file_permissions: int = 0o600  # Owner read/write only
    enable_secret_scanning: bool = True

    # Path settings with validation
    temp_dir: Optional[Path] = None
    output_dir: Path = Field(default_factory=lambda: Path("./output"))

    @field_validator('temp_dir', mode='before')
    @classmethod
    def set_temp_dir(cls, v: Optional[Path]) -> Path:
        """Set temp directory to system default if not specified."""
        if v is None:
            import tempfile
            v = Path(tempfile.gettempdir()) / "audio-extraction"
        return v

    @field_validator('temp_dir', 'output_dir')
    @classmethod
    def validate_and_create_directories(cls, v: Path) -> Path:
        """Ensure directories exist with appropriate permissions."""
        if not v.exists():
            v.mkdir(parents=True, mode=0o700, exist_ok=True)
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class ConfigurationManager:
    """Centralized configuration management with lazy loading and validation.

    Usage:
        config = ConfigurationManager()
        config.validate()  # Validate configuration

        # Access provider configuration
        if config.providers.deepgram_api_key:
            key = config.providers.deepgram_api_key.get_secret_value()

        # Access security configuration
        temp_dir = config.security.temp_dir
    """

    def __init__(self, env_file: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            env_file: Path to .env file (defaults to .env in current directory)
        """
        self.env_file = env_file or Path(".env")
        self._provider_config: Optional[ProviderConfig] = None
        self._security_config: Optional[SecurityConfig] = None

    @property
    def providers(self) -> ProviderConfig:
        """Lazy-load provider configuration."""
        if self._provider_config is None:
            if self.env_file.exists():
                self._provider_config = ProviderConfig(_env_file=str(self.env_file))
            else:
                self._provider_config = ProviderConfig()
        return self._provider_config

    @property
    def security(self) -> SecurityConfig:
        """Lazy-load security configuration."""
        if self._security_config is None:
            if self.env_file.exists():
                self._security_config = SecurityConfig(_env_file=str(self.env_file))
            else:
                self._security_config = SecurityConfig()
        return self._security_config

    def validate(self) -> None:
        """Validate configuration and raise clear errors on failure.

        Raises:
            ConfigurationError: If configuration is invalid or incomplete
        """
        # Trigger lazy loading and validation
        try:
            _ = self.providers
            _ = self.security
        except ValueError as e:
            raise ConfigurationError(
                f"Configuration validation failed: {e}. "
                f"See .env.example for configuration template."
            ) from e

        # Custom validation: at least one API key required
        if not self.providers.deepgram_api_key and not self.providers.elevenlabs_api_key:
            # Allow missing keys in test environments
            if os.getenv('PYTEST_CURRENT_TEST') is None:
                raise ConfigurationError(
                    "At least one API key required (DEEPGRAM_API_KEY or ELEVENLABS_API_KEY). "
                    "Set environment variables or create .env file. "
                    "See .env.example for configuration template."
                )

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider.

        Args:
            provider: Provider name ('deepgram' or 'elevenlabs')

        Returns:
            API key string or None if not configured
        """
        if provider.lower() == 'deepgram':
            key = self.providers.deepgram_api_key
            return key.get_secret_value() if key else None
        elif provider.lower() == 'elevenlabs':
            key = self.providers.elevenlabs_api_key
            return key.get_secret_value() if key else None
        else:
            raise ValueError(f"Unknown provider: {provider}")
