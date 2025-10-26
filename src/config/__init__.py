"""Simplified configuration management using environment variables."""
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _parse_bool(value: str | bool | None) -> bool:
    """Parse boolean value from various formats."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on", "enabled")
    return bool(value)


def _parse_list(value: str | List[str] | None, delimiter: str = ",") -> List[str]:
    """Parse list value from string or return as-is if already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(delimiter) if item.strip()]
    return [] if value is None else [value]


def _getenv(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _getenv_int(key: str, default: int) -> int:
    """Get integer environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Parsed integer value

    Raises:
        ValueError: If value cannot be parsed as integer
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(
            f"Invalid integer value for {key}='{value}'. "
            f"Expected integer, got: {value}"
        ) from e


def _getenv_float(key: str, default: float) -> float:
    """Get float environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Parsed float value

    Raises:
        ValueError: If value cannot be parsed as float
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(
            f"Invalid float value for {key}='{value}'. "
            f"Expected float, got: {value}"
        ) from e


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # ========== Application Settings ==========
    app_name: str = field(default_factory=lambda: _getenv("APP_NAME", "audio-extraction-analysis"))
    app_version: str = field(default_factory=lambda: _getenv("APP_VERSION", "1.0.0"))
    environment: str = field(default_factory=lambda: _getenv("ENVIRONMENT", "production"))

    # ========== Paths ==========
    data_dir: Path = field(default_factory=lambda: Path(_getenv("DATA_DIR", "./data")))
    cache_dir: Path = field(default_factory=lambda: Path(_getenv("CACHE_DIR", "./cache")))
    temp_dir: Path = field(default_factory=lambda: Path(_getenv("TEMP_DIR", "/tmp")))

    # ========== File Handling ==========
    max_file_size: int = field(default_factory=lambda: _getenv_int("MAX_FILE_SIZE", 100000000))
    allowed_extensions: List[str] = field(
        default_factory=lambda: _parse_list(_getenv("ALLOWED_EXTENSIONS", ".mp3,.wav,.m4a,.flac,.ogg,.aac"))
    )

    # ========== Logging ==========
    log_level: str = field(default_factory=lambda: _getenv("LOG_LEVEL", "INFO").upper())
    log_format: str = field(
        default_factory=lambda: _getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    log_file: Optional[str] = field(default_factory=lambda: _getenv("LOG_FILE") or None)
    log_to_console: bool = field(default_factory=lambda: _parse_bool(_getenv("LOG_TO_CONSOLE", "true")))

    # ========== Provider Settings ==========
    default_provider: str = field(default_factory=lambda: _getenv("DEFAULT_TRANSCRIPTION_PROVIDER", "deepgram"))
    fallback_providers: List[str] = field(
        default_factory=lambda: _parse_list(_getenv("FALLBACK_PROVIDERS", "elevenlabs,whisper"))
    )

    # ========== Language Settings ==========
    default_language: str = field(default_factory=lambda: _getenv("DEFAULT_LANGUAGE", "en"))
    supported_languages: List[str] = field(
        default_factory=lambda: _parse_list(_getenv("SUPPORTED_LANGUAGES", "en,es,fr,de,it,pt,ru,ja,ko,zh"))
    )

    # ========== Feature Flags ==========
    enable_caching: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_CACHING", "true")))
    enable_retries: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_RETRIES", "true")))
    enable_health_checks: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_HEALTH_CHECKS", "true")))
    enable_metrics: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_METRICS", "false")))

    # ========== API Keys ==========
    DEEPGRAM_API_KEY: Optional[str] = field(default_factory=lambda: _getenv("DEEPGRAM_API_KEY") or None)
    ELEVENLABS_API_KEY: Optional[str] = field(default_factory=lambda: _getenv("ELEVENLABS_API_KEY") or None)
    GEMINI_API_KEY: Optional[str] = field(default_factory=lambda: _getenv("GEMINI_API_KEY") or None)
    OPENAI_API_KEY: Optional[str] = field(default_factory=lambda: _getenv("OPENAI_API_KEY") or None)
    ANTHROPIC_API_KEY: Optional[str] = field(default_factory=lambda: _getenv("ANTHROPIC_API_KEY") or None)

    # ========== Security Settings ==========
    enable_api_key_validation: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_API_KEY_VALIDATION", "true")))
    enable_rate_limiting: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_RATE_LIMITING", "true")))
    enable_input_sanitization: bool = field(default_factory=lambda: _parse_bool(_getenv("ENABLE_INPUT_SANITIZATION", "true")))
    rate_limit_window: int = field(default_factory=lambda: _getenv_int("RATE_LIMIT_WINDOW", 60))
    rate_limit_max_requests: int = field(default_factory=lambda: _getenv_int("RATE_LIMIT_MAX_REQUESTS", 100))
    ssl_verify: bool = field(default_factory=lambda: _parse_bool(_getenv("SSL_VERIFY", "true")))
    request_timeout: int = field(default_factory=lambda: _getenv_int("REQUEST_TIMEOUT", 30))

    # ========== Performance Settings ==========
    max_workers: int = field(default_factory=lambda: _getenv_int("MAX_WORKERS", 4))
    max_concurrent_requests: int = field(default_factory=lambda: _getenv_int("MAX_CONCURRENT_REQUESTS", 10))
    thread_pool_size: int = field(default_factory=lambda: _getenv_int("THREAD_POOL_SIZE", 10))
    process_pool_size: int = field(default_factory=lambda: _getenv_int("PROCESS_POOL_SIZE", 4))

    # ========== Timeout Settings ==========
    global_timeout: int = field(default_factory=lambda: _getenv_int("GLOBAL_TIMEOUT", 600))
    connect_timeout: int = field(default_factory=lambda: _getenv_int("CONNECT_TIMEOUT", 10))
    read_timeout: int = field(default_factory=lambda: _getenv_int("READ_TIMEOUT", 30))
    write_timeout: int = field(default_factory=lambda: _getenv_int("WRITE_TIMEOUT", 30))

    # ========== Retry Settings ==========
    max_retries: int = field(default_factory=lambda: _getenv_int("MAX_API_RETRIES", 3))
    retry_delay: float = field(default_factory=lambda: _getenv_float("API_RETRY_DELAY", 1.0))
    max_retry_delay: float = field(default_factory=lambda: _getenv_float("MAX_RETRY_DELAY", 60.0))
    retry_exponential_base: float = field(default_factory=lambda: _getenv_float("RETRY_EXPONENTIAL_BASE", 2.0))
    retry_jitter: bool = field(default_factory=lambda: _parse_bool(_getenv("RETRY_JITTER_ENABLED", "true")))

    # ========== Circuit Breaker Settings ==========
    circuit_breaker_enabled: bool = field(default_factory=lambda: _parse_bool(_getenv("CIRCUIT_BREAKER_ENABLED", "true")))
    circuit_breaker_failure_threshold: int = field(default_factory=lambda: _getenv_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5))
    circuit_breaker_recovery_timeout: float = field(default_factory=lambda: _getenv_float("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 60.0))

    # ========== Batch Processing ==========
    batch_size: int = field(default_factory=lambda: _getenv_int("BATCH_SIZE", 5))
    batch_timeout: int = field(default_factory=lambda: _getenv_int("BATCH_TIMEOUT", 60))

    # ========== Caching ==========
    cache_ttl: int = field(default_factory=lambda: _getenv_int("CACHE_TTL", 3600))
    cache_max_size: int = field(default_factory=lambda: _getenv_int("CACHE_MAX_SIZE", 1000))

    # ========== UI Settings ==========
    output_format: str = field(default_factory=lambda: _getenv("OUTPUT_FORMAT", "text").lower())
    verbose: bool = field(default_factory=lambda: _parse_bool(_getenv("VERBOSE", "false")))
    quiet: bool = field(default_factory=lambda: _parse_bool(_getenv("QUIET", "false")))
    debug: bool = field(default_factory=lambda: _parse_bool(_getenv("DEBUG", "false")))
    no_color: bool = field(default_factory=lambda: _parse_bool(_getenv("NO_COLOR", "false")) or os.getenv("NO_COLOR") is not None)
    show_progress: bool = field(default_factory=lambda: _parse_bool(_getenv("SHOW_PROGRESS", "true")))
    rich_output: bool = field(default_factory=lambda: _parse_bool(_getenv("RICH_OUTPUT", "true")))

    # ========== Markdown Settings ==========
    markdown_include_timestamps: bool = field(default_factory=lambda: _parse_bool(_getenv("MARKDOWN_INCLUDE_TIMESTAMPS", "true")))
    markdown_include_speakers: bool = field(default_factory=lambda: _parse_bool(_getenv("MARKDOWN_INCLUDE_SPEAKERS", "true")))
    markdown_include_confidence: bool = field(default_factory=lambda: _parse_bool(_getenv("MARKDOWN_INCLUDE_CONFIDENCE", "false")))
    markdown_template: str = field(default_factory=lambda: _getenv("MARKDOWN_TEMPLATE", "default"))

    # ========== Deepgram Settings ==========
    DEEPGRAM_MODEL: str = field(default_factory=lambda: _getenv("DEEPGRAM_MODEL", "nova-2"))
    DEEPGRAM_LANGUAGE: str = field(default_factory=lambda: _getenv("DEEPGRAM_LANGUAGE", "en"))
    DEEPGRAM_TIMEOUT: int = field(default_factory=lambda: int(_getenv("DEEPGRAM_TIMEOUT", "600")))
    DEEPGRAM_PUNCTUATE: bool = field(default_factory=lambda: _parse_bool(_getenv("DEEPGRAM_PUNCTUATE", "true")))
    DEEPGRAM_DIARIZE: bool = field(default_factory=lambda: _parse_bool(_getenv("DEEPGRAM_DIARIZE", "false")))
    DEEPGRAM_SMART_FORMAT: bool = field(default_factory=lambda: _parse_bool(_getenv("DEEPGRAM_SMART_FORMAT", "true")))

    # ========== ElevenLabs Settings ==========
    ELEVENLABS_MODEL: str = field(default_factory=lambda: _getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"))
    ELEVENLABS_TIMEOUT: int = field(default_factory=lambda: int(_getenv("ELEVENLABS_TIMEOUT", "600")))

    # ========== Whisper Settings ==========
    WHISPER_MODEL: str = field(default_factory=lambda: _getenv("WHISPER_MODEL", "base"))
    WHISPER_DEVICE: str = field(default_factory=lambda: _getenv("WHISPER_DEVICE", "cpu"))
    WHISPER_COMPUTE_TYPE: str = field(default_factory=lambda: _getenv("WHISPER_COMPUTE_TYPE", "int8"))
    WHISPER_TIMEOUT: int = field(default_factory=lambda: int(_getenv("WHISPER_TIMEOUT", "600")))

    # ========== Parakeet Settings ==========
    PARAKEET_MODEL: str = field(default_factory=lambda: _getenv("PARAKEET_MODEL", "stt_en_conformer_ctc_large"))
    PARAKEET_DEVICE: str = field(default_factory=lambda: _getenv("PARAKEET_DEVICE", "cpu"))
    PARAKEET_BATCH_SIZE: int = field(default_factory=lambda: int(_getenv("PARAKEET_BATCH_SIZE", "1")))
    PARAKEET_USE_FP16: bool = field(default_factory=lambda: _parse_bool(_getenv("PARAKEET_USE_FP16", "false")))

    def __post_init__(self):
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Return repr with redacted API keys for security."""
        sensitive_fields = {
            'DEEPGRAM_API_KEY', 'ELEVENLABS_API_KEY', 'GEMINI_API_KEY',
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'
        }
        items = []
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name in sensitive_fields and value:
                items.append(f"{field_name}='***REDACTED***'")
            else:
                items.append(f"{field_name}={value!r}")
        return f"Config({', '.join(items)})"

    # ========== Backward Compatibility Properties ==========

    @property
    def DEFAULT_TRANSCRIPTION_PROVIDER(self) -> str:
        """Get default transcription provider."""
        return self.default_provider

    @property
    def AVAILABLE_PROVIDERS(self) -> List[str]:
        """Get available providers."""
        return ["deepgram", "elevenlabs", "whisper", "parakeet", "auto"]

    @property
    def DEFAULT_LANGUAGE(self) -> str:
        """Get default language."""
        return self.default_language

    @property
    def MAX_FILE_SIZE(self) -> int:
        """Get maximum file size."""
        return self.max_file_size

    @property
    def ALLOWED_FILE_EXTENSIONS(self) -> set:
        """Get allowed file extensions."""
        return set(self.allowed_extensions)

    @property
    def MAX_API_RETRIES(self) -> int:
        """Get maximum API retries."""
        return self.max_retries

    @property
    def API_RETRY_DELAY(self) -> float:
        """Get API retry delay."""
        return self.retry_delay

    @property
    def MAX_RETRY_DELAY(self) -> float:
        """Get maximum retry delay."""
        return self.max_retry_delay

    @property
    def RETRY_EXPONENTIAL_BASE(self) -> float:
        """Get retry exponential base."""
        return self.retry_exponential_base

    @property
    def RETRY_JITTER_ENABLED(self) -> bool:
        """Get retry jitter setting."""
        return self.retry_jitter

    @property
    def CIRCUIT_BREAKER_FAILURE_THRESHOLD(self) -> int:
        """Get circuit breaker failure threshold."""
        return self.circuit_breaker_failure_threshold

    @property
    def CIRCUIT_BREAKER_RECOVERY_TIMEOUT(self) -> float:
        """Get circuit breaker recovery timeout."""
        return self.circuit_breaker_recovery_timeout

    @property
    def HEALTH_CHECK_TIMEOUT(self) -> float:
        """Get health check timeout."""
        return float(self.connect_timeout)

    @property
    def HEALTH_CHECK_ENABLED(self) -> bool:
        """Get health check enabled setting."""
        return self.enable_health_checks

    @property
    def LOG_LEVEL(self) -> str:
        """Get log level."""
        return self.log_level

    @property
    def markdown_default_template(self) -> str:
        """Get markdown template."""
        return self.markdown_template

    # ========== Class Methods for Backward Compatibility ==========

    @classmethod
    def validate(cls, provider: str = "deepgram") -> None:
        """Validate configuration for specified provider."""
        config = cls()

        if provider == "deepgram":
            if not config.DEEPGRAM_API_KEY:
                raise ValueError(
                    "DEEPGRAM_API_KEY environment variable not found or invalid. "
                    "Set it in your environment or create a .env file with: "
                    "DEEPGRAM_API_KEY=your-api-key-here"
                )
        elif provider == "elevenlabs":
            if not config.ELEVENLABS_API_KEY:
                raise ValueError(
                    "ELEVENLABS_API_KEY environment variable not found or invalid. "
                    "Set it in your environment or create a .env file with: "
                    "ELEVENLABS_API_KEY=your-api-key-here"
                )
        elif provider == "whisper":
            try:
                import torch
                import whisper
            except ImportError:
                raise ValueError(
                    "Whisper dependencies not installed. Install with: "
                    "pip install openai-whisper torch"
                )
        elif provider == "parakeet":
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
        """Get Deepgram API key with validation."""
        config = cls()
        if not config.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY not configured")
        return config.DEEPGRAM_API_KEY

    @classmethod
    def get_elevenlabs_api_key(cls) -> str:
        """Get ElevenLabs API key with validation."""
        config = cls()
        if not config.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY not configured")
        return config.ELEVENLABS_API_KEY

    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key with validation."""
        config = cls()
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        return config.GEMINI_API_KEY

    @classmethod
    def is_configured(cls, provider: Optional[str] = None) -> bool:
        """Check if provider is configured."""
        config = cls()

        if provider == "deepgram":
            return config.DEEPGRAM_API_KEY is not None
        elif provider == "elevenlabs":
            return config.ELEVENLABS_API_KEY is not None
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
            return config.DEEPGRAM_API_KEY is not None or config.ELEVENLABS_API_KEY is not None

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of configured providers."""
        config = cls()
        available = []

        if config.DEEPGRAM_API_KEY:
            available.append("deepgram")
        if config.ELEVENLABS_API_KEY:
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
    def validate_file_extension(cls, file_path: Path) -> bool:
        """Validate file extension."""
        config = cls()
        return file_path.suffix.lower() in config.allowed_extensions


# Singleton instance with thread-safe initialization
_config_instance: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get global config instance (singleton pattern, thread-safe)."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            # Double-check pattern to prevent race conditions
            if _config_instance is None:
                _config_instance = Config()
    return _config_instance


# For backward compatibility with __init__.py exports
__all__ = ["Config", "get_config"]
