"""ElevenLabs provider configuration."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

from ..base import BaseConfig, ConfigurationSchema
from ..performance import get_performance_config
from ..security import get_security_config

logger = logging.getLogger(__name__)


class ElevenLabsModel(Enum):
    """Available ElevenLabs models."""

    MULTILINGUAL_V2 = "eleven_multilingual_v2"
    MULTILINGUAL_V1 = "eleven_multilingual_v1"
    MONOLINGUAL_V1 = "eleven_monolingual_v1"
    TURBO_V2 = "eleven_turbo_v2"


class ElevenLabsVoice(Enum):
    """Common ElevenLabs voices."""

    RACHEL = "21m00Tcm4TlvDq8ikWAM"
    DOMI = "AZnzlk1XvdvUeBnXmlld"
    BELLA = "EXAVITQu4vr4xnSDxMaL"
    ANTONI = "ErXwobaYiN019PkySvjV"
    ELLI = "MF3mGyEYCl7XYWbV9V6O"
    JOSH = "TxGEqnHWrfWFTfGW9XjX"
    ARNOLD = "VR6AewLTigWG4xSOukaG"
    ADAM = "pNInz6obpgDQGcFmaJgB"
    SAM = "yoZ06aMxZJJ28mfd3POQ"


class ElevenLabsConfig(BaseConfig):
    """ElevenLabs-specific configuration."""

    def __init__(self):
        """Initialize ElevenLabs configuration."""
        super().__init__()

        # Get dependencies
        self._security = get_security_config()
        self._performance = get_performance_config()

        # API settings
        self.api_key = self._get_api_key()
        self.api_url = self.get_value("ELEVENLABS_API_URL", "https://api.elevenlabs.io")
        self.api_version = self.get_value("ELEVENLABS_API_VERSION", "v1")

        # Model settings
        model_str = self.get_value("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.model = self._parse_model(model_str)

        # Voice settings for TTS (if using speech synthesis)
        voice_str = self.get_value("ELEVENLABS_VOICE_ID")
        self.voice_id = self._parse_voice(voice_str)
        self.voice_settings = {
            "stability": float(self.get_value("ELEVENLABS_VOICE_STABILITY", "0.5")),
            "similarity_boost": float(self.get_value("ELEVENLABS_VOICE_SIMILARITY", "0.75")),
            "style": float(self.get_value("ELEVENLABS_VOICE_STYLE", "0.0")),
            "use_speaker_boost": self.parse_bool(
                self.get_value("ELEVENLABS_SPEAKER_BOOST", "true")
            ),
        }

        # Audio settings
        self.output_format = self.get_value("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
        self.sample_rate = int(self.get_value("ELEVENLABS_SAMPLE_RATE", "44100"))
        self.bitrate = self.get_value("ELEVENLABS_BITRATE", "128k")

        # Streaming settings
        self.stream_chunk_size = int(self.get_value("ELEVENLABS_STREAM_CHUNK_SIZE", "1024"))
        self.optimize_streaming_latency = int(
            self.get_value("ELEVENLABS_OPTIMIZE_STREAMING_LATENCY", "0")
        )

        # Performance settings
        self.timeout = int(self.get_value("ELEVENLABS_TIMEOUT", "300"))
        self.max_file_size = int(self.get_value("ELEVENLABS_MAX_FILE_SIZE", "52428800"))  # 50MB
        self.rate_limit = int(self.get_value("ELEVENLABS_RATE_LIMIT", "50"))  # requests per minute

        # Advanced settings
        self.enable_ssml = self.parse_bool(self.get_value("ELEVENLABS_ENABLE_SSML", "false"))
        self.enable_phonemes = self.parse_bool(
            self.get_value("ELEVENLABS_ENABLE_PHONEMES", "false")
        )

        # Pronunciation dictionary
        self.pronunciation_dict_id = self.get_value("ELEVENLABS_PRONUNCIATION_DICT_ID")
        self.pronunciation_dict_version = self.get_value(
            "ELEVENLABS_PRONUNCIATION_DICT_VERSION", "latest"
        )

        # Projects and history
        self.project_id = self.get_value("ELEVENLABS_PROJECT_ID")
        self.save_history = self.parse_bool(self.get_value("ELEVENLABS_SAVE_HISTORY", "true"))

        # Dubbing settings (for video dubbing features)
        self.dubbing_source_lang = self.get_value("ELEVENLABS_DUBBING_SOURCE_LANG", "auto")
        self.dubbing_target_lang = self.get_value("ELEVENLABS_DUBBING_TARGET_LANG", "en")
        self.dubbing_num_speakers = int(self.get_value("ELEVENLABS_DUBBING_NUM_SPEAKERS", "0"))
        self.dubbing_watermark = self.parse_bool(
            self.get_value("ELEVENLABS_DUBBING_WATERMARK", "false")
        )

        # Voice cloning settings
        self.voice_clone_name = self.get_value("ELEVENLABS_VOICE_CLONE_NAME")
        self.voice_clone_description = self.get_value("ELEVENLABS_VOICE_CLONE_DESCRIPTION")
        self.voice_clone_labels = self.parse_list(
            self.get_value("ELEVENLABS_VOICE_CLONE_LABELS", "")
        )

        # Webhook settings
        self.webhook_url = self.get_value("ELEVENLABS_WEBHOOK_URL")
        self.webhook_events = self.parse_list(
            self.get_value("ELEVENLABS_WEBHOOK_EVENTS", "job.completed,job.failed")
        )

        # Usage and quotas
        self.character_limit = int(self.get_value("ELEVENLABS_CHARACTER_LIMIT", "100000"))
        self.enable_quota_check = self.parse_bool(
            self.get_value("ELEVENLABS_ENABLE_QUOTA_CHECK", "true")
        )

        # Metrics and monitoring
        self.enable_metrics = self.parse_bool(self.get_value("ELEVENLABS_ENABLE_METRICS", "false"))
        self.metrics_interval = int(self.get_value("ELEVENLABS_METRICS_INTERVAL", "60"))

    def _get_api_key(self) -> Optional[str]:
        """Get ElevenLabs API key from security config.

        Returns:
            API key or None
        """
        try:
            return self._security.get_api_key("elevenlabs")
        except ValueError:
            logger.warning("ElevenLabs API key not configured")
            return None

    def _parse_model(self, model_str: Optional[str]) -> ElevenLabsModel:
        """Parse model string to enum.

        Args:
            model_str: Model string

        Returns:
            ElevenLabsModel enum value
        """
        if not model_str:
            return ElevenLabsModel.MULTILINGUAL_V2

        try:
            return ElevenLabsModel(model_str)
        except ValueError:
            logger.warning(f"Unknown ElevenLabs model: {model_str}, using default")
            return ElevenLabsModel.MULTILINGUAL_V2

    def _parse_voice(self, voice_str: Optional[str]) -> Optional[str]:
        """Parse voice string to ID.

        Args:
            voice_str: Voice string (name or ID)

        Returns:
            Voice ID or None
        """
        if not voice_str:
            return None

        # Check if it's a known voice name
        try:
            voice_enum = ElevenLabsVoice[voice_str.upper()]
            return voice_enum.value
        except (KeyError, AttributeError):
            # Assume it's a custom voice ID
            return voice_str

    def get_tts_options(self) -> Dict[str, Any]:
        """Get text-to-speech options for ElevenLabs API.

        Returns:
            Dictionary of TTS options
        """
        options = {"model_id": self.model.value, "voice_settings": self.voice_settings}

        if self.pronunciation_dict_id:
            options["pronunciation_dictionary_locators"] = [
                {
                    "pronunciation_dictionary_id": self.pronunciation_dict_id,
                    "version_id": self.pronunciation_dict_version,
                }
            ]

        if self.optimize_streaming_latency:
            options["optimize_streaming_latency"] = self.optimize_streaming_latency

        if self.output_format:
            options["output_format"] = self.output_format

        return options

    def get_dubbing_options(self) -> Dict[str, Any]:
        """Get dubbing options for ElevenLabs API.

        Returns:
            Dictionary of dubbing options
        """
        options = {"source_lang": self.dubbing_source_lang, "target_lang": self.dubbing_target_lang}

        if self.dubbing_num_speakers > 0:
            options["num_speakers"] = self.dubbing_num_speakers

        if self.dubbing_watermark:
            options["watermark"] = True

        if self.project_id:
            options["project_id"] = self.project_id

        return options

    def get_api_headers(self) -> Dict[str, str]:
        """Get API headers for ElevenLabs requests.

        Returns:
            Dictionary of headers
        """
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "audio-extraction-analysis/1.0",
        }

        return headers

    def get_request_config(self) -> Dict[str, Any]:
        """Get request configuration for HTTP client.

        Returns:
            Request configuration dictionary
        """
        return {
            "base_url": f"{self.api_url}/{self.api_version}",
            "headers": self.get_api_headers(),
            "timeout": self.timeout,
            "verify_ssl": self._security.ssl_verify,
            **self._performance.get_retry_config(),
        }

    def get_streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration.

        Returns:
            Streaming configuration dictionary
        """
        return {
            "chunk_size": self.stream_chunk_size,
            "optimize_latency": self.optimize_streaming_latency,
            "output_format": self.output_format,
        }

    def estimate_cost(self, characters: int) -> float:
        """Estimate TTS cost based on character count.

        Args:
            characters: Number of characters

        Returns:
            Estimated cost in USD
        """
        # Approximate pricing (varies by plan)
        # Free tier: 10,000 characters/month
        # Starter: $5/month for 30,000 characters
        # Creator: $22/month for 100,000 characters
        # Professional: $99/month for 500,000 characters

        cost_per_1k_chars = 0.30  # Approximate for pay-as-you-go

        return (characters / 1000) * cost_per_1k_chars

    def check_quota(self) -> Optional[Dict[str, Any]]:
        """Check usage quota if enabled.

        Returns:
            Quota information or None
        """
        if not self.enable_quota_check:
            return None

        # This would make an API call to check quota
        # For now, return mock data
        return {
            "character_count": 0,
            "character_limit": self.character_limit,
            "remaining": self.character_limit,
        }

    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size against ElevenLabs limits.

        Args:
            file_size: File size in bytes

        Returns:
            True if valid, False otherwise
        """
        return file_size <= self.max_file_size

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for ElevenLabs config
        """
        return ConfigurationSchema(
            name="ElevenLabsConfig",
            required_fields={"api_key"},
            optional_fields={
                "model": "eleven_multilingual_v2",
                "output_format": "mp3_44100_128",
                "timeout": 300,
                "save_history": True,
            },
            validators={
                "model": lambda x: x in {m.value for m in ElevenLabsModel},
                "timeout": lambda x: isinstance(x, int) and x > 0,
                "sample_rate": lambda x: isinstance(x, int)
                and x in {8000, 11025, 16000, 22050, 24000, 44100, 48000},
                "voice_stability": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
                "voice_similarity": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary (with sensitive data redacted)
        """
        return {
            "api_url": self.api_url,
            "api_version": self.api_version,
            "model": self.model.value,
            "voice_id": self.voice_id,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "timeout": self.timeout,
            "max_file_size": self.max_file_size,
            "save_history": self.save_history,
            "enable_quota_check": self.enable_quota_check,
            "character_limit": self.character_limit,
            "api_key_configured": self.api_key is not None,
        }


# Singleton instance getter
def get_elevenlabs_config() -> ElevenLabsConfig:
    """Get ElevenLabs configuration instance.

    Returns:
        ElevenLabsConfig singleton instance
    """
    return ElevenLabsConfig.get_instance()
