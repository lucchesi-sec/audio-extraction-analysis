"""Comprehensive tests for refactored configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    Config,
    ConfigPriority,
    GlobalConfig,
    PerformanceConfig,
    UIConfig,
    ValidationLevel,
    create_config_validator,
    get_global_config,
    get_performance_config,
    get_security_config,
    get_ui_config,
)
from src.config.providers import DeepgramConfig, ElevenLabsConfig, ParakeetConfig, WhisperConfig


class TestBackwardCompatibility:
    """Test backward compatibility with existing Config class."""

    def test_config_class_exists(self):
        """Test that Config class still exists."""
        config = Config()
        assert config is not None

    def test_api_key_properties(self):
        """Test API key properties work."""
        with patch.dict(
            os.environ,
            {
                "DEEPGRAM_API_KEY": "test_deepgram_key_1234567890abcdef1234567890abcdef12345678",
                "ELEVENLABS_API_KEY": "test_elevenlabs_key_1234567890ab",
                "GEMINI_API_KEY": "AIzaTestKey123456789012345678901234567",
            },
        ):
            config = Config()

            # These should return the mocked values (without validation)
            assert config.DEEPGRAM_API_KEY is not None
            assert config.ELEVENLABS_API_KEY is not None
            assert config.GEMINI_API_KEY is not None

    def test_provider_settings(self):
        """Test provider configuration properties."""
        config = Config()

        # Should have default values
        assert config.DEFAULT_TRANSCRIPTION_PROVIDER == "deepgram"
        assert "deepgram" in config.AVAILABLE_PROVIDERS
        assert config.DEFAULT_LANGUAGE == "en"
        assert config.MAX_FILE_SIZE > 0

    def test_performance_settings(self):
        """Test performance configuration properties."""
        config = Config()

        assert config.MAX_API_RETRIES >= 1
        assert config.API_RETRY_DELAY > 0
        assert config.MAX_RETRY_DELAY >= config.API_RETRY_DELAY
        assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD > 0

    def test_class_methods(self):
        """Test backward compatible class methods."""
        # Test is_configured
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key"}):
            assert Config.is_configured("deepgram") is False  # Invalid key format

        # Test validate_file_extension
        assert Config.validate_file_extension(Path("test.mp3")) is True
        assert Config.validate_file_extension(Path("test.exe")) is False

    def test_get_provider_config(self):
        """Test getting provider configuration."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "a" * 40}):
            config = Config.get_provider_config("deepgram")

            assert "max_retries" in config
            assert "timeout" in config
            assert "max_file_size" in config


class TestGlobalConfig:
    """Test global configuration module."""

    def test_singleton_pattern(self):
        """Test singleton implementation."""
        config1 = get_global_config()
        config2 = get_global_config()
        assert config1 is config2

    def test_environment_loading(self):
        """Test environment variable loading."""
        with patch.dict(
            os.environ, {"APP_NAME": "test_app", "MAX_FILE_SIZE": "50000000", "LOG_LEVEL": "DEBUG"}
        ):
            config = GlobalConfig()
            assert config.app_name == "test_app"
            assert config.max_file_size == 50000000
            assert config.log_level == "DEBUG"

    def test_directory_creation(self):
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ, {"DATA_DIR": f"{tmpdir}/data", "CACHE_DIR": f"{tmpdir}/cache"}
            ):
                GlobalConfig()
                assert Path(f"{tmpdir}/data").exists()
                assert Path(f"{tmpdir}/cache").exists()

    def test_config_overlay(self):
        """Test configuration overlay system."""
        config = get_global_config()

        # Set overlay at different priorities
        config.set_overlay(ConfigPriority.DEFAULTS, {"test_key": "default"})
        config.set_overlay(ConfigPriority.FILE, {"test_key": "file"})
        config.set_overlay(ConfigPriority.CLI, {"test_key": "cli"})

        # CLI should win
        assert config.get_value("test_key") == "cli"

    def test_validation(self):
        """Test configuration validation."""
        config = get_global_config()

        # Should validate successfully with default values
        assert config.validate() is True

    def test_parse_helpers(self):
        """Test parsing helper methods."""
        config = get_global_config()

        assert config.parse_bool("true") is True
        assert config.parse_bool("false") is False
        assert config.parse_bool("1") is True
        assert config.parse_bool("0") is False

        assert config.parse_list("a,b,c") == ["a", "b", "c"]
        assert config.parse_list("") == []


class TestSecurityConfig:
    """Test security configuration module."""

    def test_api_key_validation(self):
        """Test API key format validation."""
        config = get_security_config()

        # Valid formats
        assert config.validate_api_key("deepgram", "a" * 40) is True
        assert config.validate_api_key("elevenlabs", "b" * 32) is True
        assert config.validate_api_key("gemini", "AIza" + "x" * 35) is True

        # Invalid formats
        assert config.validate_api_key("deepgram", "short") is False
        assert config.validate_api_key("elevenlabs", "wrong_length") is False

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = get_security_config()
        config.enable_rate_limiting = True
        config.rate_limit_max_requests = 5
        config.rate_limit_window = 1  # 1 second window

        identifier = "test_user"

        # Should allow initial requests
        for _ in range(5):
            assert config.check_rate_limit(identifier) is True

        # Should block after limit
        assert config.check_rate_limit(identifier, increment=False) is False

    def test_path_sanitization(self):
        """Test path sanitization."""
        config = get_security_config()
        config.enable_input_sanitization = True

        # Valid paths
        valid_path = Path("/tmp/test.txt")
        assert config.sanitize_path(valid_path) == valid_path.resolve()

        # Should reject dangerous patterns
        config.blocked_path_patterns = ["../", "/etc/"]

        with pytest.raises(ValueError):
            config.sanitize_path(Path("../etc/passwd"))

    def test_sensitive_data_handling(self):
        """Test sensitive data sanitization."""
        config = get_security_config()

        # Test sanitization for logging
        api_key = "secret_api_key_12345678"
        sanitized = config.sanitize_for_logging(api_key)
        assert "secret" not in sanitized
        assert "****" in sanitized

        # Test hashing
        data = "sensitive_data"
        hash1 = config.hash_sensitive_data(data)
        hash2 = config.hash_sensitive_data(data)
        assert hash1 == hash2  # Consistent hashing
        assert data not in hash1  # Original not in hash


class TestPerformanceConfig:
    """Test performance configuration module."""

    def test_performance_profiles(self):
        """Test predefined performance profiles."""
        with patch.dict(os.environ, {"PERFORMANCE_PROFILE": "high"}):
            config = PerformanceConfig()
            assert config.profile.name == "high"
            assert config.profile.max_workers > 4

    def test_retry_configuration(self):
        """Test retry settings."""
        config = get_performance_config()
        retry_config = config.get_retry_config()

        assert "max_attempts" in retry_config
        assert "initial_delay" in retry_config
        assert "exponential_base" in retry_config
        assert retry_config["max_attempts"] > 0

    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        config = get_performance_config()
        cb_config = config.get_circuit_breaker_config()

        assert cb_config["enabled"] is True
        assert cb_config["failure_threshold"] > 0
        assert cb_config["recovery_timeout"] > 0

    def test_resource_tracking(self):
        """Test resource usage tracking."""
        config = get_performance_config()

        # Update resource usage
        config.update_resource_usage("cpu", 0.5)
        config.update_resource_usage("memory", 1024)

        usage = config.get_resource_usage()
        assert usage["cpu"] == 0.5
        assert usage["memory"] == 1024

        # Test throttling
        config.cpu_limit = 0.4
        assert config.should_throttle() is True

    def test_optimal_batch_size(self):
        """Test dynamic batch size calculation."""
        config = get_performance_config()
        config.batch_size = 10

        # Normal conditions
        assert config.get_optimal_batch_size() == 10

        # Throttled conditions
        config.cpu_limit = 0.1
        config.update_resource_usage("cpu", 0.2)
        assert config.get_optimal_batch_size() < 10


class TestUIConfig:
    """Test UI configuration module."""

    def test_output_formats(self):
        """Test output format configuration."""
        with patch.dict(os.environ, {"OUTPUT_FORMAT": "json"}):
            config = UIConfig()
            assert config.output_format.value == "json"

    def test_color_detection(self):
        """Test color output detection."""
        config = get_ui_config()

        # Test NO_COLOR standard
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert config.should_use_color() is False

        # Test force color
        config.force_color = True
        assert config.should_use_color() is True

    def test_progress_configuration(self):
        """Test progress bar configuration."""
        config = get_ui_config()
        progress_config = config.get_progress_config()

        assert "style" in progress_config
        assert "width" in progress_config
        assert "refresh_rate" in progress_config
        assert progress_config["width"] > 0

    def test_format_output(self):
        """Test output formatting."""
        config = get_ui_config()

        data = {"key": "value", "number": 42}

        # Test JSON formatting
        from src.config.ui import OutputFormat

        json_output = config.format_output(data, OutputFormat.JSON)
        assert '"key"' in json_output
        assert '"value"' in json_output


class TestProviderConfigs:
    """Test provider-specific configurations."""

    def test_deepgram_config(self):
        """Test Deepgram configuration."""
        with patch.dict(
            os.environ,
            {"DEEPGRAM_API_KEY": "a" * 40, "DEEPGRAM_MODEL": "nova-2", "DEEPGRAM_LANGUAGE": "en"},
        ):
            config = DeepgramConfig()

            assert config.model.value == "nova-2"
            assert config.language == "en"

            # Test transcription options
            options = config.get_transcription_options()
            assert options["model"] == "nova-2"
            assert options["language"] == "en"

    def test_whisper_config(self):
        """Test Whisper configuration."""
        with patch.dict(os.environ, {"WHISPER_MODEL": "base", "WHISPER_DEVICE": "cpu"}):
            config = WhisperConfig()

            assert config.model.value == "base"
            assert config.device.value == "cpu"

            # Test memory estimation
            memory = config.estimate_memory_usage()
            assert memory > 0

    def test_elevenlabs_config(self):
        """Test ElevenLabs configuration."""
        with patch.dict(
            os.environ,
            {"ELEVENLABS_API_KEY": "b" * 32, "ELEVENLABS_MODEL": "eleven_multilingual_v2"},
        ):
            config = ElevenLabsConfig()

            assert config.model.value == "eleven_multilingual_v2"

            # Test TTS options
            options = config.get_tts_options()
            assert "model_id" in options
            assert "voice_settings" in options

    def test_parakeet_config(self):
        """Test Parakeet configuration."""
        with patch.dict(
            os.environ,
            {"PARAKEET_MODEL": "stt_en_conformer_ctc_large", "PARAKEET_BATCH_SIZE": "16"},
        ):
            config = ParakeetConfig()

            assert config.model.value == "stt_en_conformer_ctc_large"
            assert config.batch_size == 16

            # Test model configuration
            model_config = config.get_model_config()
            assert model_config["model_name"] == "stt_en_conformer_ctc_large"
            assert model_config["batch_size"] == 16


class TestConfigurationValidation:
    """Test configuration validation system."""

    def test_validator_creation(self):
        """Test validator creation with different levels."""
        validator_strict = create_config_validator(ValidationLevel.STRICT)
        validator_normal = create_config_validator(ValidationLevel.NORMAL)
        validator_lenient = create_config_validator(ValidationLevel.LENIENT)

        assert validator_strict.level == ValidationLevel.STRICT
        assert validator_normal.level == ValidationLevel.NORMAL
        assert validator_lenient.level == ValidationLevel.LENIENT

    def test_validation_rules(self):
        """Test validation rule application."""
        validator = create_config_validator(ValidationLevel.NORMAL)

        config = {
            "api_key": "test_key_with_sufficient_length_12345",
            "timeout": 30,
            "port": 8080,
            "file_path": "/tmp/test.txt",
        }

        # Should pass basic validation
        assert validator.validate(config) is True

        # Test with invalid values
        invalid_config = {"port": 70000, "timeout": -1}  # Out of range  # Negative

        assert validator.validate(invalid_config) is False
        assert len(validator.errors) > 0

    def test_auto_fix(self):
        """Test automatic value correction."""
        validator = create_config_validator(ValidationLevel.NORMAL)

        config = {"timeout": 5000}  # Too large

        # Validator should auto-fix if not in strict mode
        validator.validate(config)

        # Check for warnings about auto-fix
        assert len(validator.warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
