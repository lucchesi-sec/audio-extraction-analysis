"""Factory for creating transcription service providers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..config.config import Config
from ..utils.retry import RetryConfig
from .base import BaseTranscriptionProvider, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class TranscriptionProviderFactory:
    """Factory class for creating and managing transcription service providers."""

    # Registry of available providers
    _providers: Dict[str, Type[BaseTranscriptionProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseTranscriptionProvider]) -> None:
        """Register a transcription provider.

        Args:
            name: Provider name (e.g., 'deepgram', 'elevenlabs')
            provider_class: Provider class implementing BaseTranscriptionProvider
        """
        cls._providers[name] = provider_class
        logger.debug(f"Registered transcription provider: {name}")

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names that are registered
        """
        return list(cls._providers.keys())

    @classmethod
    def get_configured_providers(cls) -> List[str]:
        """Get list of providers that have valid API keys configured.

        Returns:
            List of provider names that are properly configured
        """
        configured = []

        # Check Deepgram
        if Config.DEEPGRAM_API_KEY:
            configured.append("deepgram")

        # Check ElevenLabs
        if Config.ELEVENLABS_API_KEY:
            configured.append("elevenlabs")

        # Check Whisper (always available if dependencies installed)
        try:
            import torch
            import whisper

            configured.append("whisper")
        except (ImportError, Exception):
            pass

        # Check Parakeet (always available if dependencies installed)
        try:
            import nemo.collections.asr as nemo_asr

            configured.append("parakeet")
        except (ImportError, Exception):
            pass

        return configured

    @classmethod
    def _get_default_configs(
        cls,
        provider_name: str,
        circuit_config: Optional[CircuitBreakerConfig],
        retry_config: Optional[RetryConfig],
    ) -> tuple[CircuitBreakerConfig, RetryConfig]:
        """Get default circuit breaker and retry configurations.

        Args:
            provider_name: Name of the provider
            circuit_config: Existing circuit config or None
            retry_config: Existing retry config or None

        Returns:
            Tuple of (circuit_config, retry_config)
        """
        if circuit_config is not None and retry_config is not None:
            return circuit_config, retry_config

        config_dict = Config.get_provider_config(provider_name)

        if circuit_config is None:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=config_dict.get("circuit_breaker_failure_threshold", 5),
                recovery_timeout=config_dict.get("circuit_breaker_recovery_timeout", 60.0),
            )

        if retry_config is None:
            retry_config = RetryConfig(
                max_attempts=config_dict.get("max_retries", 3),
                base_delay=config_dict.get("retry_delay", 1.0),
                max_delay=config_dict.get("max_retry_delay", 60.0),
                exponential_base=config_dict.get("retry_exponential_base", 2.0),
                jitter=config_dict.get("retry_jitter_enabled", True),
            )

        return circuit_config, retry_config

    @classmethod
    def _run_health_check(cls, provider: BaseTranscriptionProvider, provider_name: str) -> None:
        """Run health check on provider and log results.

        Args:
            provider: Provider instance to check
            provider_name: Name of the provider for logging
        """
        if not Config.HEALTH_CHECK_ENABLED:
            return

        try:
            health_result = provider.health_check()
            if not health_result.get("healthy", False):
                logger.warning(
                    f"Provider '{provider_name}' health check failed: {health_result.get('status')}"
                )
                # Don't raise error, just log warning - provider might recover
            else:
                logger.info(f"Provider '{provider_name}' health check passed")
        except Exception as e:
            logger.warning(f"Health check failed for '{provider_name}': {e}")
            # Don't raise error - health check is informational

    @classmethod
    def _create_provider_instance(
        cls,
        provider_class: Type[BaseTranscriptionProvider],
        provider_name: str,
        api_key: Optional[str],
        circuit_config: CircuitBreakerConfig,
        retry_config: RetryConfig,
    ) -> BaseTranscriptionProvider:
        """Create and validate provider instance.

        Args:
            provider_class: Provider class to instantiate
            provider_name: Name of the provider for error messages
            api_key: Optional API key
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration

        Returns:
            Validated provider instance

        Raises:
            ValueError: If provider configuration is invalid
        """
        provider = provider_class(
            api_key=api_key, circuit_config=circuit_config, retry_config=retry_config
        )

        if not provider.validate_configuration():
            raise ValueError(f"Provider '{provider_name}' is not properly configured")

        return provider

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        run_health_check: bool = True,
    ) -> BaseTranscriptionProvider:
        """Create a transcription provider instance.

        Args:
            provider_name: Name of the provider to create ('deepgram', 'elevenlabs')
            api_key: Optional API key. If None, uses configuration
            circuit_config: Optional circuit breaker configuration
            retry_config: Optional retry configuration
            run_health_check: Whether to run health check after creation

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not registered or not configured
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls.get_available_providers())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]

        try:
            # Get default configurations if not provided
            circuit_config, retry_config = cls._get_default_configs(
                provider_name, circuit_config, retry_config
            )

            # Create and validate provider instance
            provider = cls._create_provider_instance(
                provider_class, provider_name, api_key, circuit_config, retry_config
            )

            # Run health check if requested
            if run_health_check:
                cls._run_health_check(provider, provider_name)

            logger.info(f"Created transcription provider: {provider.get_provider_name()}")
            return provider

        except ValueError as e:
            logger.error(f"Failed to create provider '{provider_name}': {e}")
            raise
        except ImportError as e:
            logger.error(f"Provider module not available '{provider_name}': {e}")
            raise ValueError(f"Provider '{provider_name}' module not available") from e
        except Exception as e:
            logger.error(f"Unexpected error creating provider '{provider_name}': {e}")
            raise ValueError(f"Failed to create provider '{provider_name}'") from e

    @classmethod
    def auto_select_provider(
        cls,
        audio_file_path: Optional[Path] = None,
        preferred_features: Optional[List[str]] = None,
        include_health_check: bool = True,
    ) -> str:
        """Automatically select the best available provider.

        Args:
            audio_file_path: Optional path to audio file for size-based selection
            preferred_features: Optional list of required features
            include_health_check: Whether to consider health status in selection

        Returns:
            Name of the selected provider

        Raises:
            ValueError: If no providers are configured
        """
        configured_providers = cls.get_configured_providers()

        if not configured_providers:
            raise ValueError(
                "No transcription providers are configured. "
                "Please set DEEPGRAM_API_KEY or ELEVENLABS_API_KEY environment variables."
            )

        # Filter by health status if requested
        if include_health_check and Config.HEALTH_CHECK_ENABLED:
            healthy_providers = []
            for provider_name in configured_providers:
                try:
                    health = cls.check_provider_health_sync(provider_name)
                    if health.get("healthy", False):
                        healthy_providers.append(provider_name)
                    else:
                        logger.warning(
                            f"Provider '{provider_name}' failed health check: {health.get('status')}"
                        )
                except Exception as e:
                    logger.warning(f"Health check failed for '{provider_name}': {e}")

            if healthy_providers:
                configured_providers = healthy_providers
                logger.info(f"Filtered to healthy providers: {configured_providers}")
            else:
                logger.warning("No providers passed health check, using all configured providers")

        # If only one provider is available, use it
        if len(configured_providers) == 1:
            selected = configured_providers[0]
            logger.info(f"Auto-selected provider (only available): {selected}")
            return selected

        # Check file size constraints if audio file provided
        if audio_file_path and audio_file_path.exists():
            file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)

            # ElevenLabs has 50MB limit, Deepgram supports larger files
            if file_size_mb > 50:
                if "deepgram" in configured_providers:
                    logger.info(
                        f"Auto-selected provider (file size {file_size_mb:.1f}MB): deepgram"
                    )
                    return "deepgram"
                else:
                    raise ValueError(
                        f"File size {file_size_mb:.1f}MB exceeds limits of available providers"
                    )

        # Check feature requirements
        if preferred_features:
            best_provider = None
            max_features = 0

            for provider_name in configured_providers:
                try:
                    provider = cls.create_provider(provider_name, run_health_check=False)
                    supported_features = provider.get_supported_features()
                    matching_features = len(set(preferred_features) & set(supported_features))

                    if matching_features > max_features:
                        max_features = matching_features
                        best_provider = provider_name

                except Exception as e:
                    logger.debug(f"Could not evaluate features for '{provider_name}': {e}")
                    continue

            if best_provider:
                logger.info(f"Auto-selected provider (feature matching): {best_provider}")
                return best_provider

        # Default priority: Deepgram > ElevenLabs > Whisper > Parakeet (based on feature richness)
        priority_order = ["deepgram", "elevenlabs", "whisper", "parakeet"]

        for provider_name in priority_order:
            if provider_name in configured_providers:
                logger.info(f"Auto-selected provider (default priority): {provider_name}")
                return provider_name

        # Fallback to first available provider
        selected = configured_providers[0]
        logger.info(f"Auto-selected provider (fallback): {selected}")
        return selected

    @classmethod
    def validate_provider_for_file(cls, provider_name: str, audio_file_path: Path) -> bool:
        """Validate that a provider can handle the given audio file.

        Args:
            provider_name: Name of the provider
            audio_file_path: Path to audio file

        Returns:
            True if provider can handle the file, False otherwise
        """
        if not audio_file_path.exists():
            return False

        file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)

        # Check provider-specific constraints
        if provider_name == "elevenlabs":
            if file_size_mb > 50:  # ElevenLabs 50MB limit
                logger.warning(f"File size {file_size_mb:.1f}MB exceeds ElevenLabs 50MB limit")
                return False
        elif provider_name == "deepgram":
            if file_size_mb > 2000:  # Deepgram ~2GB limit (conservative)
                logger.warning(f"File size {file_size_mb:.1f}MB exceeds Deepgram limit")
                return False
        elif provider_name == "whisper":
            # Whisper can handle large files, but check global file size limit
            if file_size_mb > (Config.MAX_FILE_SIZE / (1024 * 1024)):
                logger.warning(f"File size {file_size_mb:.1f}MB exceeds global file size limit")
                return False
        elif provider_name == "parakeet":
            # Parakeet can handle large files, but check global file size limit
            if file_size_mb > (Config.MAX_FILE_SIZE / (1024 * 1024)):
                logger.warning(f"File size {file_size_mb:.1f}MB exceeds global file size limit")
                return False

        return True

    @classmethod
    async def check_provider_health(
        cls, provider_name: str, api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check health of a specific provider without creating full instance.

        Args:
            provider_name: Name of the provider to check
            api_key: Optional API key

        Returns:
            Dictionary containing health check results

        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls.get_available_providers())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available providers: {available}"
            )

        try:
            provider = cls.create_provider(provider_name, api_key, run_health_check=False)
            return await provider.health_check_async()
        except Exception as e:
            return {
                "healthy": False,
                "status": "creation_failed",
                "response_time_ms": 0,
                "details": {
                    "provider": provider_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }

    @classmethod
    def check_provider_health_sync(
        cls, provider_name: str, api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous version of provider health check.

        Args:
            provider_name: Name of the provider to check
            api_key: Optional API key

        Returns:
            Dictionary containing health check results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(cls.check_provider_health(provider_name, api_key))
        finally:
            if not loop.is_running():
                loop.close()

    @classmethod
    def get_provider_status(cls) -> Dict[str, Any]:
        """Get status of all configured providers.

        Returns:
            Dictionary containing status of all providers
        """
        status = {
            "available_providers": cls.get_available_providers(),
            "configured_providers": cls.get_configured_providers(),
            "provider_health": {},
        }

        # Check health of configured providers
        for provider_name in cls.get_configured_providers():
            try:
                health = cls.check_provider_health_sync(provider_name)
                status["provider_health"][provider_name] = health
            except Exception as e:
                status["provider_health"][provider_name] = {
                    "healthy": False,
                    "status": "check_failed",
                    "error": str(e),
                }

        return status


# Initialize the factory with default providers
def _initialize_factory():
    """Initialize factory with available providers."""
    try:
        from .deepgram import DeepgramTranscriber

        TranscriptionProviderFactory.register_provider("deepgram", DeepgramTranscriber)
        logger.debug("Registered Deepgram provider")
    except ImportError as e:
        logger.warning(f"Deepgram provider not available: {e}")

    try:
        from .elevenlabs import ElevenLabsTranscriber

        TranscriptionProviderFactory.register_provider("elevenlabs", ElevenLabsTranscriber)
        logger.debug("Registered ElevenLabs provider")
    except ImportError as e:
        logger.warning(f"ElevenLabs provider not available: {e}")

    try:
        from .whisper import WhisperTranscriber

        TranscriptionProviderFactory.register_provider("whisper", WhisperTranscriber)
        logger.debug("Registered Whisper provider")
    except ImportError as e:
        logger.warning(f"Whisper provider not available: {e}")

    try:
        from .parakeet import ParakeetTranscriber

        TranscriptionProviderFactory.register_provider("parakeet", ParakeetTranscriber)
        logger.debug("Registered Parakeet provider")
    except (ImportError, Exception) as e:
        logger.warning(f"Parakeet provider not available: {e}")


# Initialize providers when module is imported
_initialize_factory()
