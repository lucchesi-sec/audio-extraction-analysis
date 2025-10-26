"""Comprehensive test suite for src/config/__init__.py module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config, _getenv, _parse_bool, _parse_list, get_config


class TestHelperFunctions:
    """Test helper functions for configuration parsing."""

    # Tests for _parse_bool
    def test_parse_bool_with_boolean_true(self):
        """Test _parse_bool with boolean True."""
        assert _parse_bool(True) is True

    def test_parse_bool_with_boolean_false(self):
        """Test _parse_bool with boolean False."""
        assert _parse_bool(False) is False

    def test_parse_bool_with_none(self):
        """Test _parse_bool with None returns False."""
        assert _parse_bool(None) is False

    def test_parse_bool_with_string_true(self):
        """Test _parse_bool with string 'true'."""
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True

    def test_parse_bool_with_string_false(self):
        """Test _parse_bool with string 'false'."""
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("FALSE") is False

    def test_parse_bool_with_string_numeric(self):
        """Test _parse_bool with numeric strings."""
        assert _parse_bool("1") is True
        assert _parse_bool("0") is False

    def test_parse_bool_with_string_yes(self):
        """Test _parse_bool with 'yes'."""
        assert _parse_bool("yes") is True
        assert _parse_bool("Yes") is True
        assert _parse_bool("YES") is True

    def test_parse_bool_with_string_on(self):
        """Test _parse_bool with 'on'."""
        assert _parse_bool("on") is True
        assert _parse_bool("On") is True
        assert _parse_bool("ON") is True

    def test_parse_bool_with_string_enabled(self):
        """Test _parse_bool with 'enabled'."""
        assert _parse_bool("enabled") is True
        assert _parse_bool("Enabled") is True

    def test_parse_bool_with_invalid_string(self):
        """Test _parse_bool with invalid string returns False."""
        assert _parse_bool("invalid") is False
        assert _parse_bool("no") is False
        assert _parse_bool("off") is False
        assert _parse_bool("disabled") is False

    def test_parse_bool_with_empty_string(self):
        """Test _parse_bool with empty string."""
        assert _parse_bool("") is False

    def test_parse_bool_with_numeric_values(self):
        """Test _parse_bool with numeric values."""
        assert _parse_bool(1) is True
        assert _parse_bool(0) is False
        assert _parse_bool(42) is True

    # Tests for _parse_list
    def test_parse_list_with_list(self):
        """Test _parse_list with list returns as-is."""
        input_list = ["a", "b", "c"]
        assert _parse_list(input_list) == input_list

    def test_parse_list_with_string(self):
        """Test _parse_list with comma-separated string."""
        assert _parse_list("a,b,c") == ["a", "b", "c"]

    def test_parse_list_with_string_spaces(self):
        """Test _parse_list strips whitespace."""
        assert _parse_list("a, b , c") == ["a", "b", "c"]
        assert _parse_list(" a , b , c ") == ["a", "b", "c"]

    def test_parse_list_with_none(self):
        """Test _parse_list with None returns empty list."""
        assert _parse_list(None) == []

    def test_parse_list_with_empty_string(self):
        """Test _parse_list with empty string."""
        assert _parse_list("") == []

    def test_parse_list_with_custom_delimiter(self):
        """Test _parse_list with custom delimiter."""
        assert _parse_list("a|b|c", delimiter="|") == ["a", "b", "c"]
        assert _parse_list("a:b:c", delimiter=":") == ["a", "b", "c"]

    def test_parse_list_ignores_empty_items(self):
        """Test _parse_list filters out empty items."""
        assert _parse_list("a,,b,,c") == ["a", "b", "c"]
        assert _parse_list(",a,b,c,") == ["a", "b", "c"]

    def test_parse_list_with_single_value(self):
        """Test _parse_list with single value string."""
        assert _parse_list("single") == ["single"]

    def test_parse_list_with_non_string_non_list(self):
        """Test _parse_list with non-string, non-list value."""
        assert _parse_list(42) == [42]
        assert _parse_list(True) == [True]

    # Tests for _getenv
    def test_getenv_with_existing_variable(self):
        """Test _getenv retrieves existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert _getenv("TEST_VAR") == "test_value"

    def test_getenv_with_missing_variable(self):
        """Test _getenv returns default for missing variable."""
        with patch.dict(os.environ, {}, clear=True):
            assert _getenv("MISSING_VAR", "default") == "default"

    def test_getenv_with_missing_variable_no_default(self):
        """Test _getenv returns empty string when no default provided."""
        with patch.dict(os.environ, {}, clear=True):
            assert _getenv("MISSING_VAR") == ""

    def test_getenv_with_empty_value(self):
        """Test _getenv with empty environment variable."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            assert _getenv("EMPTY_VAR") == ""

    def test_getenv_case_sensitive(self):
        """Test _getenv is case-sensitive."""
        with patch.dict(os.environ, {"test_var": "lowercase"}):
            assert _getenv("TEST_VAR", "default") == "default"
            assert _getenv("test_var") == "lowercase"


class TestConfigInitialization:
    """Test Config class initialization and defaults."""

    def test_config_initialization_defaults(self):
        """Test Config initializes with all default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            # Application settings
            assert config.app_name == "audio-extraction-analysis"
            assert config.app_version == "1.0.0"
            assert config.environment == "production"

            # Paths
            assert config.data_dir == Path("./data")
            assert config.cache_dir == Path("./cache")
            assert config.temp_dir == Path("/tmp")

            # File handling
            assert config.max_file_size == 100000000
            assert ".mp3" in config.allowed_extensions
            assert ".wav" in config.allowed_extensions

            # Logging
            assert config.log_level == "INFO"
            assert config.log_to_console is True

            # Provider settings
            assert config.default_provider == "deepgram"
            assert "elevenlabs" in config.fallback_providers

            # Language settings
            assert config.default_language == "en"
            assert "en" in config.supported_languages

            # Feature flags
            assert config.enable_caching is True
            assert config.enable_retries is True
            assert config.enable_health_checks is True
            assert config.enable_metrics is False

    def test_config_initialization_with_environment_variables(self):
        """Test Config reads from environment variables."""
        with patch.dict(os.environ, {
            "APP_NAME": "custom_app",
            "APP_VERSION": "2.0.0",
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG",
            "MAX_FILE_SIZE": "50000000",
            "DEFAULT_LANGUAGE": "es",
        }):
            config = Config()

            assert config.app_name == "custom_app"
            assert config.app_version == "2.0.0"
            assert config.environment == "development"
            assert config.log_level == "DEBUG"
            assert config.max_file_size == 50000000
            assert config.default_language == "es"

    def test_config_post_init_creates_directories(self):
        """Test __post_init__ creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            cache_path = Path(tmpdir) / "cache"

            with patch.dict(os.environ, {
                "DATA_DIR": str(data_path),
                "CACHE_DIR": str(cache_path),
            }):
                config = Config()

                assert data_path.exists()
                assert cache_path.exists()
                assert data_path.is_dir()
                assert cache_path.is_dir()

    def test_config_boolean_fields_parsing(self):
        """Test boolean fields are parsed correctly from environment."""
        with patch.dict(os.environ, {
            "LOG_TO_CONSOLE": "false",
            "ENABLE_CACHING": "0",
            "ENABLE_RETRIES": "off",
            "ENABLE_METRICS": "true",
        }):
            config = Config()

            assert config.log_to_console is False
            assert config.enable_caching is False
            assert config.enable_retries is False
            assert config.enable_metrics is True

    def test_config_list_fields_parsing(self):
        """Test list fields are parsed correctly from environment."""
        with patch.dict(os.environ, {
            "ALLOWED_EXTENSIONS": ".mp3,.wav,.flac",
            "FALLBACK_PROVIDERS": "whisper,elevenlabs",
            "SUPPORTED_LANGUAGES": "en,es,fr",
        }):
            config = Config()

            assert config.allowed_extensions == [".mp3", ".wav", ".flac"]
            assert config.fallback_providers == ["whisper", "elevenlabs"]
            assert config.supported_languages == ["en", "es", "fr"]

    def test_config_integer_fields_parsing(self):
        """Test integer fields are parsed correctly from environment."""
        with patch.dict(os.environ, {
            "MAX_FILE_SIZE": "123456",
            "MAX_WORKERS": "8",
            "MAX_API_RETRIES": "5",
            "BATCH_SIZE": "10",
        }):
            config = Config()

            assert config.max_file_size == 123456
            assert config.max_workers == 8
            assert config.max_retries == 5
            assert config.batch_size == 10

    def test_config_float_fields_parsing(self):
        """Test float fields are parsed correctly from environment."""
        with patch.dict(os.environ, {
            "API_RETRY_DELAY": "2.5",
            "MAX_RETRY_DELAY": "120.0",
            "RETRY_EXPONENTIAL_BASE": "3.0",
        }):
            config = Config()

            assert config.retry_delay == 2.5
            assert config.max_retry_delay == 120.0
            assert config.retry_exponential_base == 3.0

    def test_config_api_keys_optional(self):
        """Test API keys are optional and default to None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            assert config.DEEPGRAM_API_KEY is None
            assert config.ELEVENLABS_API_KEY is None
            assert config.GEMINI_API_KEY is None
            assert config.OPENAI_API_KEY is None
            assert config.ANTHROPIC_API_KEY is None

    def test_config_api_keys_from_environment(self):
        """Test API keys are read from environment."""
        with patch.dict(os.environ, {
            "DEEPGRAM_API_KEY": "dg_key_123",
            "ELEVENLABS_API_KEY": "el_key_456",
            "GEMINI_API_KEY": "gem_key_789",
        }):
            config = Config()

            assert config.DEEPGRAM_API_KEY == "dg_key_123"
            assert config.ELEVENLABS_API_KEY == "el_key_456"
            assert config.GEMINI_API_KEY == "gem_key_789"

    def test_config_no_color_special_handling(self):
        """Test NO_COLOR environment variable special handling."""
        # NO_COLOR set to any value
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            config = Config()
            assert config.no_color is True

        # NO_COLOR set to empty string (presence is what matters)
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            config = Config()
            assert config.no_color is True

        # NO_COLOR not set
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.no_color is False


class TestConfigProperties:
    """Test Config backward compatibility properties."""

    def test_default_transcription_provider_property(self):
        """Test DEFAULT_TRANSCRIPTION_PROVIDER property."""
        with patch.dict(os.environ, {"DEFAULT_TRANSCRIPTION_PROVIDER": "whisper"}):
            config = Config()
            assert config.DEFAULT_TRANSCRIPTION_PROVIDER == "whisper"
            assert config.default_provider == "whisper"

    def test_available_providers_property(self):
        """Test AVAILABLE_PROVIDERS property."""
        config = Config()
        providers = config.AVAILABLE_PROVIDERS

        assert isinstance(providers, list)
        assert "deepgram" in providers
        assert "elevenlabs" in providers
        assert "whisper" in providers
        assert "parakeet" in providers
        assert "auto" in providers

    def test_default_language_property(self):
        """Test DEFAULT_LANGUAGE property."""
        with patch.dict(os.environ, {"DEFAULT_LANGUAGE": "fr"}):
            config = Config()
            assert config.DEFAULT_LANGUAGE == "fr"
            assert config.default_language == "fr"

    def test_max_file_size_property(self):
        """Test MAX_FILE_SIZE property."""
        with patch.dict(os.environ, {"MAX_FILE_SIZE": "50000000"}):
            config = Config()
            assert config.MAX_FILE_SIZE == 50000000
            assert config.max_file_size == 50000000

    def test_allowed_file_extensions_property(self):
        """Test ALLOWED_FILE_EXTENSIONS returns a set."""
        config = Config()
        extensions = config.ALLOWED_FILE_EXTENSIONS

        assert isinstance(extensions, set)
        assert ".mp3" in extensions
        assert ".wav" in extensions

    def test_retry_properties(self):
        """Test retry-related backward compatibility properties."""
        with patch.dict(os.environ, {
            "MAX_API_RETRIES": "5",
            "API_RETRY_DELAY": "2.0",
            "MAX_RETRY_DELAY": "120.0",
            "RETRY_EXPONENTIAL_BASE": "3.0",
            "RETRY_JITTER_ENABLED": "false",
        }):
            config = Config()

            assert config.MAX_API_RETRIES == 5
            assert config.API_RETRY_DELAY == 2.0
            assert config.MAX_RETRY_DELAY == 120.0
            assert config.RETRY_EXPONENTIAL_BASE == 3.0
            assert config.RETRY_JITTER_ENABLED is False

    def test_circuit_breaker_properties(self):
        """Test circuit breaker backward compatibility properties."""
        with patch.dict(os.environ, {
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
            "CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "120.0",
        }):
            config = Config()

            assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 10
            assert config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT == 120.0

    def test_health_check_properties(self):
        """Test health check backward compatibility properties."""
        with patch.dict(os.environ, {
            "ENABLE_HEALTH_CHECKS": "false",
            "CONNECT_TIMEOUT": "15",
        }):
            config = Config()

            assert config.HEALTH_CHECK_ENABLED is False
            assert config.HEALTH_CHECK_TIMEOUT == 15.0

    def test_log_level_property(self):
        """Test LOG_LEVEL property."""
        with patch.dict(os.environ, {"LOG_LEVEL": "warning"}):
            config = Config()
            assert config.LOG_LEVEL == "WARNING"

    def test_markdown_default_template_property(self):
        """Test markdown_default_template property."""
        with patch.dict(os.environ, {"MARKDOWN_TEMPLATE": "custom"}):
            config = Config()
            assert config.markdown_default_template == "custom"


class TestConfigClassMethods:
    """Test Config class methods."""

    def test_is_configured_deepgram_with_key(self):
        """Test is_configured for deepgram with API key."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key"}):
            assert Config.is_configured("deepgram") is True

    def test_is_configured_deepgram_without_key(self):
        """Test is_configured for deepgram without API key."""
        with patch.dict(os.environ, {}, clear=True):
            assert Config.is_configured("deepgram") is False

    def test_is_configured_elevenlabs_with_key(self):
        """Test is_configured for elevenlabs with API key."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}):
            assert Config.is_configured("elevenlabs") is True

    def test_is_configured_elevenlabs_without_key(self):
        """Test is_configured for elevenlabs without API key."""
        with patch.dict(os.environ, {}, clear=True):
            assert Config.is_configured("elevenlabs") is False

    def test_is_configured_no_provider_with_deepgram(self):
        """Test is_configured with no provider but deepgram key present."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key"}):
            assert Config.is_configured() is True

    def test_is_configured_no_provider_with_elevenlabs(self):
        """Test is_configured with no provider but elevenlabs key present."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}):
            assert Config.is_configured() is True

    def test_is_configured_no_provider_no_keys(self):
        """Test is_configured with no provider and no keys."""
        with patch.dict(os.environ, {}, clear=True):
            assert Config.is_configured() is False

    def test_get_deepgram_api_key_success(self):
        """Test get_deepgram_api_key with valid key."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "valid_key"}):
            key = Config.get_deepgram_api_key()
            assert key == "valid_key"

    def test_get_deepgram_api_key_missing(self):
        """Test get_deepgram_api_key raises error when key missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.get_deepgram_api_key()

            assert "DEEPGRAM_API_KEY not configured" in str(exc_info.value)

    def test_get_elevenlabs_api_key_success(self):
        """Test get_elevenlabs_api_key with valid key."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "valid_key"}):
            key = Config.get_elevenlabs_api_key()
            assert key == "valid_key"

    def test_get_elevenlabs_api_key_missing(self):
        """Test get_elevenlabs_api_key raises error when key missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.get_elevenlabs_api_key()

            assert "ELEVENLABS_API_KEY not configured" in str(exc_info.value)

    def test_get_gemini_api_key_success(self):
        """Test get_gemini_api_key with valid key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            key = Config.get_gemini_api_key()
            assert key == "valid_key"

    def test_get_gemini_api_key_missing(self):
        """Test get_gemini_api_key raises error when key missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.get_gemini_api_key()

            assert "GEMINI_API_KEY not configured" in str(exc_info.value)

    def test_get_available_providers_with_deepgram(self):
        """Test get_available_providers includes deepgram when key present."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key"}):
            providers = Config.get_available_providers()
            assert "deepgram" in providers

    def test_get_available_providers_with_elevenlabs(self):
        """Test get_available_providers includes elevenlabs when key present."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}):
            providers = Config.get_available_providers()
            assert "elevenlabs" in providers

    def test_get_available_providers_empty(self):
        """Test get_available_providers with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            providers = Config.get_available_providers()
            assert isinstance(providers, list)
            # May include whisper/parakeet if installed

    def test_validate_file_extension_valid(self):
        """Test validate_file_extension with valid extensions."""
        assert Config.validate_file_extension(Path("test.mp3")) is True
        assert Config.validate_file_extension(Path("test.wav")) is True
        assert Config.validate_file_extension(Path("test.m4a")) is True

    def test_validate_file_extension_invalid(self):
        """Test validate_file_extension with invalid extensions."""
        assert Config.validate_file_extension(Path("test.txt")) is False
        assert Config.validate_file_extension(Path("test.exe")) is False
        assert Config.validate_file_extension(Path("test.pdf")) is False

    def test_validate_file_extension_case_insensitive(self):
        """Test validate_file_extension is case-insensitive."""
        assert Config.validate_file_extension(Path("test.MP3")) is True
        assert Config.validate_file_extension(Path("test.WAV")) is True
        assert Config.validate_file_extension(Path("test.M4A")) is True


class TestGetConfigSingleton:
    """Test get_config singleton pattern."""

    def test_get_config_returns_config_instance(self):
        """Test get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same instance (singleton)."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_get_config_singleton_maintains_state(self):
        """Test singleton maintains state across calls."""
        # Reset singleton
        import src.config
        src.config._config_instance = None

        # First call creates instance
        config1 = get_config()
        config1.app_name = "modified_name"

        # Second call should return same instance with modified state
        config2 = get_config()
        assert config2.app_name == "modified_name"
        assert config1 is config2


class TestConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_config_with_empty_string_values(self):
        """Test Config handles empty string environment variables."""
        with patch.dict(os.environ, {
            "APP_NAME": "",
            "LOG_LEVEL": "",
            "DEFAULT_LANGUAGE": "",
        }):
            config = Config()

            assert config.app_name == ""
            assert config.log_level == ""
            assert config.default_language == ""

    def test_config_path_objects(self):
        """Test Config creates Path objects correctly."""
        config = Config()

        assert isinstance(config.data_dir, Path)
        assert isinstance(config.cache_dir, Path)
        assert isinstance(config.temp_dir, Path)

    def test_config_log_file_optional(self):
        """Test log_file is optional and defaults to None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.log_file is None

        with patch.dict(os.environ, {"LOG_FILE": ""}):
            config = Config()
            assert config.log_file is None

        with patch.dict(os.environ, {"LOG_FILE": "/var/log/app.log"}):
            config = Config()
            assert config.log_file == "/var/log/app.log"

    def test_config_provider_settings_deepgram(self):
        """Test Deepgram-specific settings."""
        with patch.dict(os.environ, {
            "DEEPGRAM_MODEL": "nova-3",
            "DEEPGRAM_LANGUAGE": "es",
            "DEEPGRAM_TIMEOUT": "300",
            "DEEPGRAM_PUNCTUATE": "false",
            "DEEPGRAM_DIARIZE": "true",
        }):
            config = Config()

            assert config.DEEPGRAM_MODEL == "nova-3"
            assert config.DEEPGRAM_LANGUAGE == "es"
            assert config.DEEPGRAM_TIMEOUT == 300
            assert config.DEEPGRAM_PUNCTUATE is False
            assert config.DEEPGRAM_DIARIZE is True

    def test_config_provider_settings_whisper(self):
        """Test Whisper-specific settings."""
        with patch.dict(os.environ, {
            "WHISPER_MODEL": "large",
            "WHISPER_DEVICE": "cuda",
            "WHISPER_COMPUTE_TYPE": "float16",
        }):
            config = Config()

            assert config.WHISPER_MODEL == "large"
            assert config.WHISPER_DEVICE == "cuda"
            assert config.WHISPER_COMPUTE_TYPE == "float16"

    def test_config_log_level_uppercase_conversion(self):
        """Test log level is converted to uppercase."""
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            config = Config()
            assert config.log_level == "DEBUG"

        with patch.dict(os.environ, {"LOG_LEVEL": "WaRnInG"}):
            config = Config()
            assert config.log_level == "WARNING"

    def test_config_all_timeouts(self):
        """Test all timeout settings."""
        with patch.dict(os.environ, {
            "GLOBAL_TIMEOUT": "1200",
            "CONNECT_TIMEOUT": "20",
            "READ_TIMEOUT": "60",
            "WRITE_TIMEOUT": "60",
            "REQUEST_TIMEOUT": "45",
        }):
            config = Config()

            assert config.global_timeout == 1200
            assert config.connect_timeout == 20
            assert config.read_timeout == 60
            assert config.write_timeout == 60
            assert config.request_timeout == 45


class TestConfigExports:
    """Test module exports."""

    def test_module_all_exports(self):
        """Test __all__ contains expected exports."""
        from src.config import __all__

        assert "Config" in __all__
        assert "get_config" in __all__
        assert len(__all__) == 2

    def test_module_imports(self):
        """Test all expected imports work."""
        from src.config import Config, get_config

        assert Config is not None
        assert get_config is not None
        assert callable(get_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
