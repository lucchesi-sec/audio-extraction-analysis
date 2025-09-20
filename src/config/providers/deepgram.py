"""Deepgram provider configuration."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

from ..base import BaseConfig, ConfigurationSchema
from ..performance import get_performance_config
from ..security import get_security_config

logger = logging.getLogger(__name__)


class DeepgramModel(Enum):
    """Available Deepgram models."""

    NOVA_2 = "nova-2"
    NOVA = "nova"
    ENHANCED = "enhanced"
    BASE = "base"
    WHISPER_LARGE = "whisper-large"
    WHISPER_MEDIUM = "whisper-medium"
    WHISPER_SMALL = "whisper-small"
    WHISPER_BASE = "whisper-base"
    WHISPER_TINY = "whisper-tiny"


class DeepgramTier(Enum):
    """Deepgram pricing tiers."""

    PAY_AS_YOU_GO = "pay-as-you-go"
    GROWTH = "growth"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class DeepgramConfig(BaseConfig):
    """Deepgram-specific configuration."""

    def __init__(self):
        """Initialize Deepgram configuration."""
        super().__init__()

        # Get dependencies
        self._security = get_security_config()
        self._performance = get_performance_config()

        # API settings
        self.api_key = self._get_api_key()
        self.api_url = self.get_value("DEEPGRAM_API_URL", "https://api.deepgram.com")
        self.api_version = self.get_value("DEEPGRAM_API_VERSION", "v1")

        # Model settings
        self.model = DeepgramModel(self.get_value("DEEPGRAM_MODEL", "nova-2").lower())
        self.language = self.get_value("DEEPGRAM_LANGUAGE", "en")
        self.detect_language = self.parse_bool(self.get_value("DEEPGRAM_DETECT_LANGUAGE", "false"))

        # Transcription settings
        self.punctuate = self.parse_bool(self.get_value("DEEPGRAM_PUNCTUATE", "true"))
        self.profanity_filter = self.parse_bool(
            self.get_value("DEEPGRAM_PROFANITY_FILTER", "false")
        )
        self.redact = self.parse_list(self.get_value("DEEPGRAM_REDACT", ""))
        self.diarize = self.parse_bool(self.get_value("DEEPGRAM_DIARIZE", "false"))
        self.diarize_version = self.get_value("DEEPGRAM_DIARIZE_VERSION", "latest")
        self.smart_format = self.parse_bool(self.get_value("DEEPGRAM_SMART_FORMAT", "true"))
        self.utterances = self.parse_bool(self.get_value("DEEPGRAM_UTTERANCES", "false"))
        self.numerals = self.parse_bool(self.get_value("DEEPGRAM_NUMERALS", "true"))

        # Audio processing
        self.channels = int(self.get_value("DEEPGRAM_CHANNELS", "1"))
        self.sample_rate = int(self.get_value("DEEPGRAM_SAMPLE_RATE", "16000"))
        self.encoding = self.get_value("DEEPGRAM_ENCODING", "linear16")
        self.multichannel = self.parse_bool(self.get_value("DEEPGRAM_MULTICHANNEL", "false"))

        # Advanced features
        self.keywords = self.parse_list(self.get_value("DEEPGRAM_KEYWORDS", ""))
        self.keyword_boost = self.parse_list(self.get_value("DEEPGRAM_KEYWORD_BOOST", ""))
        self.search = self.parse_list(self.get_value("DEEPGRAM_SEARCH", ""))
        self.replace = self.parse_list(self.get_value("DEEPGRAM_REPLACE", ""))
        self.tag = self.parse_list(self.get_value("DEEPGRAM_TAG", ""))

        # Callback and webhook settings
        self.callback = self.get_value("DEEPGRAM_CALLBACK")
        self.callback_method = self.get_value("DEEPGRAM_CALLBACK_METHOD", "POST")

        # Paragraphs and summarization
        self.paragraphs = self.parse_bool(self.get_value("DEEPGRAM_PARAGRAPHS", "false"))
        self.summarize = self.parse_bool(self.get_value("DEEPGRAM_SUMMARIZE", "false"))
        self.detect_topics = self.parse_bool(self.get_value("DEEPGRAM_DETECT_TOPICS", "false"))
        self.detect_entities = self.parse_bool(self.get_value("DEEPGRAM_DETECT_ENTITIES", "false"))
        self.sentiment = self.parse_bool(self.get_value("DEEPGRAM_SENTIMENT", "false"))

        # Streaming settings
        self.interim_results = self.parse_bool(self.get_value("DEEPGRAM_INTERIM_RESULTS", "true"))
        self.endpointing = self.parse_bool(self.get_value("DEEPGRAM_ENDPOINTING", "false"))
        self.vad_turnoff = int(self.get_value("DEEPGRAM_VAD_TURNOFF", "1000"))

        # Performance settings
        self.timeout = int(self.get_value("DEEPGRAM_TIMEOUT", "600"))
        self.max_alternatives = int(self.get_value("DEEPGRAM_MAX_ALTERNATIVES", "1"))

        # Tier and limits
        self.tier = DeepgramTier(self.get_value("DEEPGRAM_TIER", "pay-as-you-go").lower())
        self.max_file_size = int(self.get_value("DEEPGRAM_MAX_FILE_SIZE", "2147483648"))  # 2GB
        self.rate_limit = int(self.get_value("DEEPGRAM_RATE_LIMIT", "100"))  # requests per minute

        # Custom vocabulary
        self.custom_model = self.get_value("DEEPGRAM_CUSTOM_MODEL")
        self.custom_model_id = self.get_value("DEEPGRAM_CUSTOM_MODEL_ID")

        # Metrics and monitoring
        self.enable_metrics = self.parse_bool(self.get_value("DEEPGRAM_ENABLE_METRICS", "false"))
        self.metrics_interval = int(self.get_value("DEEPGRAM_METRICS_INTERVAL", "60"))

    def _get_api_key(self) -> Optional[str]:
        """Get Deepgram API key from security config.

        Returns:
            API key or None
        """
        try:
            return self._security.get_api_key("deepgram")
        except ValueError:
            logger.warning("Deepgram API key not configured")
            return None

    def get_transcription_options(self) -> Dict[str, Any]:
        """Get transcription options for Deepgram API.

        Returns:
            Dictionary of transcription options
        """
        options = {
            "model": self.model.value,
            "language": self.language,
            "punctuate": self.punctuate,
            "smart_format": self.smart_format,
            "numerals": self.numerals,
        }

        # Add optional features
        if self.detect_language:
            options["detect_language"] = True

        if self.profanity_filter:
            options["profanity_filter"] = True

        if self.redact:
            options["redact"] = self.redact

        if self.diarize:
            options["diarize"] = True
            options["diarize_version"] = self.diarize_version

        if self.utterances:
            options["utterances"] = True

        if self.paragraphs:
            options["paragraphs"] = True

        if self.summarize:
            options["summarize"] = True

        if self.detect_topics:
            options["detect_topics"] = True

        if self.detect_entities:
            options["detect_entities"] = True

        if self.sentiment:
            options["sentiment"] = True

        if self.keywords:
            options["keywords"] = self.keywords

        if self.search:
            options["search"] = self.search

        if self.replace:
            options["replace"] = self.replace

        if self.callback:
            options["callback"] = self.callback
            options["callback_method"] = self.callback_method

        if self.custom_model:
            options["model"] = self.custom_model

        if self.max_alternatives > 1:
            options["alternatives"] = self.max_alternatives

        return options

    def get_streaming_options(self) -> Dict[str, Any]:
        """Get streaming options for Deepgram API.

        Returns:
            Dictionary of streaming options
        """
        options = self.get_transcription_options()

        # Add streaming-specific options
        options.update(
            {
                "interim_results": self.interim_results,
                "endpointing": self.endpointing,
                "vad_turnoff": self.vad_turnoff,
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            }
        )

        if self.multichannel:
            options["multichannel"] = True

        return options

    def get_api_headers(self) -> Dict[str, str]:
        """Get API headers for Deepgram requests.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "audio-extraction-analysis/1.0",
        }

        # Add custom headers if needed
        if self.custom_model_id:
            headers["X-Custom-Model-ID"] = self.custom_model_id

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

    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate transcription cost based on duration.

        Args:
            duration_seconds: Audio duration in seconds

        Returns:
            Estimated cost in USD
        """
        # Pricing per minute (approximate, varies by tier and model)
        pricing = {
            DeepgramTier.PAY_AS_YOU_GO: {
                DeepgramModel.NOVA_2: 0.0043,
                DeepgramModel.NOVA: 0.0039,
                DeepgramModel.ENHANCED: 0.0145,
                DeepgramModel.BASE: 0.0125,
            },
            DeepgramTier.GROWTH: {
                DeepgramModel.NOVA_2: 0.0036,
                DeepgramModel.NOVA: 0.0033,
                DeepgramModel.ENHANCED: 0.0120,
                DeepgramModel.BASE: 0.0100,
            },
        }

        tier_pricing = pricing.get(self.tier, pricing[DeepgramTier.PAY_AS_YOU_GO])
        model_price = tier_pricing.get(self.model, 0.0043)  # Default to Nova-2 pricing

        duration_minutes = duration_seconds / 60
        base_cost = duration_minutes * model_price

        # Add cost for additional features
        feature_multiplier = 1.0
        if self.diarize:
            feature_multiplier += 0.25
        if self.summarize:
            feature_multiplier += 0.30
        if self.sentiment:
            feature_multiplier += 0.20
        if self.detect_entities:
            feature_multiplier += 0.20

        return base_cost * feature_multiplier

    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size against Deepgram limits.

        Args:
            file_size: File size in bytes

        Returns:
            True if valid, False otherwise
        """
        return file_size <= self.max_file_size

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for Deepgram config
        """
        return ConfigurationSchema(
            name="DeepgramConfig",
            required_fields={"api_key"},
            optional_fields={
                "model": "nova-2",
                "language": "en",
                "punctuate": True,
                "smart_format": True,
                "timeout": 600,
            },
            validators={
                "model": lambda x: x in {m.value for m in DeepgramModel},
                "language": lambda x: len(x) == 2 or x == "auto",
                "timeout": lambda x: isinstance(x, int) and x > 0,
                "channels": lambda x: isinstance(x, int) and 1 <= x <= 32,
                "sample_rate": lambda x: isinstance(x, int) and x in {8000, 16000, 24000, 48000},
                "max_alternatives": lambda x: isinstance(x, int) and 1 <= x <= 25,
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
            "language": self.language,
            "detect_language": self.detect_language,
            "punctuate": self.punctuate,
            "smart_format": self.smart_format,
            "diarize": self.diarize,
            "summarize": self.summarize,
            "timeout": self.timeout,
            "tier": self.tier.value,
            "max_file_size": self.max_file_size,
            "api_key_configured": self.api_key is not None,
        }


# Singleton instance getter
def get_deepgram_config() -> DeepgramConfig:
    """Get Deepgram configuration instance.

    Returns:
        DeepgramConfig singleton instance
    """
    return DeepgramConfig.get_instance()
