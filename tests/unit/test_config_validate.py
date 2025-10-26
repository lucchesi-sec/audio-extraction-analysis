"""Comprehensive tests for Config.validate() method."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config


class TestConfigValidate:
    """Test suite for Config.validate() class method."""

    def test_validate_deepgram_success(self):
        """Test successful validation with valid Deepgram API key."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "valid_deepgram_key_123"}):
            # Should not raise any exception
            Config.validate("deepgram")

    def test_validate_deepgram_missing_key(self):
        """Test validation fails when Deepgram API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("deepgram")

            assert "DEEPGRAM_API_KEY" in str(exc_info.value)
            assert "environment variable not found" in str(exc_info.value)
            assert ".env file" in str(exc_info.value)

    def test_validate_deepgram_empty_key(self):
        """Test validation fails when Deepgram API key is empty."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": ""}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("deepgram")

            assert "DEEPGRAM_API_KEY" in str(exc_info.value)

    def test_validate_elevenlabs_success(self):
        """Test successful validation with valid ElevenLabs API key."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "valid_elevenlabs_key_456"}):
            # Should not raise any exception
            Config.validate("elevenlabs")

    def test_validate_elevenlabs_missing_key(self):
        """Test validation fails when ElevenLabs API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("elevenlabs")

            assert "ELEVENLABS_API_KEY" in str(exc_info.value)
            assert "environment variable not found" in str(exc_info.value)
            assert ".env file" in str(exc_info.value)

    def test_validate_elevenlabs_empty_key(self):
        """Test validation fails when ElevenLabs API key is empty."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("elevenlabs")

            assert "ELEVENLABS_API_KEY" in str(exc_info.value)

    def test_validate_whisper_success(self):
        """Test successful validation when Whisper dependencies are available."""
        # Mock successful imports
        mock_torch = MagicMock()
        mock_whisper = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "whisper": mock_whisper}):
            # Should not raise any exception
            Config.validate("whisper")

    def test_validate_whisper_missing_torch(self):
        """Test validation fails when torch dependency is missing."""
        with patch.dict("sys.modules", {"torch": None, "whisper": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("whisper")

            assert "Whisper dependencies not installed" in str(exc_info.value)
            assert "openai-whisper" in str(exc_info.value)
            assert "torch" in str(exc_info.value)

    def test_validate_whisper_missing_whisper(self):
        """Test validation fails when whisper dependency is missing."""
        mock_torch = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "whisper": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("whisper")

            assert "Whisper dependencies not installed" in str(exc_info.value)

    def test_validate_whisper_import_error(self):
        """Test validation fails when import raises ImportError."""
        # Simulate ImportError by removing from sys.modules
        import sys

        # Save original modules if they exist
        torch_backup = sys.modules.get("torch")
        whisper_backup = sys.modules.get("whisper")

        try:
            # Remove modules to force ImportError
            if "torch" in sys.modules:
                del sys.modules["torch"]
            if "whisper" in sys.modules:
                del sys.modules["whisper"]

            with pytest.raises(ValueError) as exc_info:
                Config.validate("whisper")

            assert "Whisper dependencies not installed" in str(exc_info.value)
        finally:
            # Restore original modules
            if torch_backup:
                sys.modules["torch"] = torch_backup
            if whisper_backup:
                sys.modules["whisper"] = whisper_backup

    def test_validate_parakeet_success(self):
        """Test successful validation when Parakeet dependencies are available."""
        mock_nemo = MagicMock()
        mock_torch = MagicMock()

        with patch.dict("sys.modules", {"nemo": mock_nemo, "torch": mock_torch}):
            # Should not raise any exception
            Config.validate("parakeet")

    def test_validate_parakeet_missing_nemo(self):
        """Test validation fails when nemo dependency is missing."""
        with patch.dict("sys.modules", {"nemo": None, "torch": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("parakeet")

            assert "Parakeet dependencies not installed" in str(exc_info.value)
            assert "nemo_toolkit" in str(exc_info.value)
            assert "torch" in str(exc_info.value)

    def test_validate_parakeet_missing_torch(self):
        """Test validation fails when torch dependency is missing for Parakeet."""
        mock_nemo = MagicMock()

        with patch.dict("sys.modules", {"nemo": mock_nemo, "torch": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("parakeet")

            assert "Parakeet dependencies not installed" in str(exc_info.value)

    def test_validate_parakeet_import_error(self):
        """Test validation fails when Parakeet import raises ImportError."""
        import sys

        # Save original modules if they exist
        nemo_backup = sys.modules.get("nemo")
        torch_backup = sys.modules.get("torch")

        try:
            # Remove modules to force ImportError
            if "nemo" in sys.modules:
                del sys.modules["nemo"]
            if "torch" in sys.modules:
                del sys.modules["torch"]

            with pytest.raises(ValueError) as exc_info:
                Config.validate("parakeet")

            assert "Parakeet dependencies not installed" in str(exc_info.value)
        finally:
            # Restore original modules
            if nemo_backup:
                sys.modules["nemo"] = nemo_backup
            if torch_backup:
                sys.modules["torch"] = torch_backup

    def test_validate_unknown_provider(self):
        """Test validation fails with unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            Config.validate("invalid_provider")

        assert "Unknown provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_validate_empty_provider(self):
        """Test validation fails with empty provider string."""
        with pytest.raises(ValueError) as exc_info:
            Config.validate("")

        assert "Unknown provider" in str(exc_info.value)

    def test_validate_none_provider_uses_default(self):
        """Test that not providing provider uses default (deepgram)."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key"}):
            # Should validate as deepgram by default
            Config.validate()  # No provider specified

    def test_validate_default_provider_no_key(self):
        """Test default provider validation fails without key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.validate()  # Uses default "deepgram"

            assert "DEEPGRAM_API_KEY" in str(exc_info.value)

    def test_validate_case_sensitivity(self):
        """Test provider names are case-sensitive."""
        # Uppercase should fail
        with pytest.raises(ValueError) as exc_info:
            Config.validate("DEEPGRAM")

        assert "Unknown provider" in str(exc_info.value)

    def test_validate_whitespace_provider(self):
        """Test provider with whitespace fails."""
        with pytest.raises(ValueError) as exc_info:
            Config.validate(" deepgram ")

        assert "Unknown provider" in str(exc_info.value)

    def test_validate_creates_new_config_instance(self):
        """Test that validate creates a fresh Config instance each time."""
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "key1"}):
            Config.validate("deepgram")

        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "key2"}):
            # Should create new instance and pick up new env var
            Config.validate("deepgram")

    def test_validate_all_providers_sequentially(self):
        """Test validating all providers in sequence."""
        # Setup all required API keys and dependencies
        mock_torch = MagicMock()
        mock_whisper = MagicMock()
        mock_nemo = MagicMock()

        with patch.dict(
            os.environ,
            {
                "DEEPGRAM_API_KEY": "deepgram_key",
                "ELEVENLABS_API_KEY": "elevenlabs_key",
            },
        ):
            with patch.dict(
                "sys.modules",
                {"torch": mock_torch, "whisper": mock_whisper, "nemo": mock_nemo},
            ):
                # All should validate successfully
                Config.validate("deepgram")
                Config.validate("elevenlabs")
                Config.validate("whisper")
                Config.validate("parakeet")


class TestConfigValidateEdgeCases:
    """Test edge cases and error message quality for Config.validate()."""

    def test_validate_error_message_includes_helpful_instructions_deepgram(self):
        """Test Deepgram error message provides clear instructions."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("deepgram")

            error_msg = str(exc_info.value)
            # Should mention environment variable
            assert "environment variable" in error_msg.lower()
            # Should suggest .env file
            assert ".env" in error_msg
            # Should show example format
            assert "DEEPGRAM_API_KEY=your-api-key-here" in error_msg

    def test_validate_error_message_includes_helpful_instructions_elevenlabs(self):
        """Test ElevenLabs error message provides clear instructions."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("elevenlabs")

            error_msg = str(exc_info.value)
            assert "environment variable" in error_msg.lower()
            assert ".env" in error_msg
            assert "ELEVENLABS_API_KEY=your-api-key-here" in error_msg

    def test_validate_error_message_includes_install_instructions_whisper(self):
        """Test Whisper error message includes pip install command."""
        with patch.dict("sys.modules", {"torch": None, "whisper": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("whisper")

            error_msg = str(exc_info.value)
            assert "pip install" in error_msg
            assert "openai-whisper" in error_msg
            assert "torch" in error_msg

    def test_validate_error_message_includes_install_instructions_parakeet(self):
        """Test Parakeet error message includes pip install command."""
        with patch.dict("sys.modules", {"nemo": None, "torch": None}):
            with pytest.raises(ValueError) as exc_info:
                Config.validate("parakeet")

            error_msg = str(exc_info.value)
            assert "pip install" in error_msg
            assert "nemo_toolkit" in error_msg
            assert "torch" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
