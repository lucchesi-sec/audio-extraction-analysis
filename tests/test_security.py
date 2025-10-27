"""Security validation tests.

This module tests security-critical functionality:
- Temporary file security (permissions, cleanup)
- API key management (no hardcoded secrets, masking)
- Configuration validation
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from src.config.secure_config import ConfigurationError, ConfigurationManager, SecurityConfig
from src.utils.secure_temp import secure_temp_file, validate_temp_file_security


class TestTempFileSecurity:
    """Test secure temporary file handling."""

    def test_secure_temp_file_permissions(self, tmp_path: Path):
        """Verify temp files created with restrictive permissions (0600)."""
        with secure_temp_file(suffix=".mp3", dir=tmp_path) as temp_file:
            assert temp_file.exists()

            # Check permissions are owner read/write only
            mode = temp_file.stat().st_mode
            # 0600 = owner read/write, no group/other permissions
            assert (mode & stat.S_IRUSR) != 0  # Owner can read
            assert (mode & stat.S_IWUSR) != 0  # Owner can write
            assert (mode & stat.S_IRGRP) == 0  # Group cannot read
            assert (mode & stat.S_IWGRP) == 0  # Group cannot write
            assert (mode & stat.S_IROTH) == 0  # Others cannot read
            assert (mode & stat.S_IWOTH) == 0  # Others cannot write

    def test_secure_temp_file_cleanup(self, tmp_path: Path):
        """Verify temp files are cleaned up after use."""
        temp_path = None

        with secure_temp_file(suffix=".mp3", dir=tmp_path) as temp_file:
            temp_path = temp_file
            assert temp_file.exists()

        # After context exit, file should be deleted
        assert not temp_path.exists()

    def test_secure_temp_file_cleanup_on_exception(self, tmp_path: Path):
        """Verify temp files cleaned up even when exception occurs."""
        temp_path = None

        try:
            with secure_temp_file(suffix=".mp3", dir=tmp_path) as temp_file:
                temp_path = temp_file
                assert temp_file.exists()
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # File should still be cleaned up
        assert not temp_path.exists()

    def test_validate_temp_file_security_pass(self, tmp_path: Path):
        """Test security validation passes for secure files."""
        with secure_temp_file(dir=tmp_path) as temp_file:
            # Write some data
            temp_file.write_text("test")

            # Should pass security validation
            assert validate_temp_file_security(temp_file)

    def test_validate_temp_file_security_fail(self, tmp_path: Path):
        """Test security validation fails for insecure files."""
        # Create file with insecure permissions
        insecure_file = tmp_path / "insecure.txt"
        insecure_file.touch()
        insecure_file.chmod(0o666)  # World-readable/writable

        # Should fail security validation
        assert not validate_temp_file_security(insecure_file)

    def test_no_predictable_filenames(self, tmp_path: Path):
        """Verify temp files don't have predictable names."""
        paths = []

        for _ in range(3):
            with secure_temp_file(prefix="test-", dir=tmp_path) as temp_file:
                paths.append(temp_file.name)

        # All filenames should be different
        assert len(set(paths)) == 3


class TestAPIKeyManagement:
    """Test secure API key handling."""

    def test_api_key_masking_in_repr(self, test_config):
        """Verify API keys are masked in string representation."""
        config = test_config

        # Get string representation
        providers_str = str(config.providers)

        # Should not contain actual API key
        assert "test-key-deepgram" not in providers_str
        assert "test-key-elevenlabs" not in providers_str

        # Should contain masking indicator
        assert "***" in providers_str or "SecretStr" in providers_str

    def test_api_key_retrieval(self, test_config):
        """Test API keys can be retrieved when needed."""
        config = test_config

        # Should be able to get the actual key value
        deepgram_key = config.get_api_key('deepgram')
        elevenlabs_key = config.get_api_key('elevenlabs')

        assert deepgram_key == "test-key-deepgram-12345678"
        assert elevenlabs_key == "test-key-elevenlabs-12345678"

    def test_invalid_api_key_rejected(self, tmp_path: Path):
        """Test API keys that are too short are rejected."""
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPGRAM_API_KEY=short\n")

        os.environ['PYTEST_CURRENT_TEST'] = 'test'
        try:
            config = ConfigurationManager(env_file=env_file)
            with pytest.raises(Exception):  # Should raise validation error
                _ = config.providers.deepgram_api_key
        finally:
            if 'PYTEST_CURRENT_TEST' in os.environ:
                del os.environ['PYTEST_CURRENT_TEST']

    def test_missing_api_keys_handled(self, tmp_path: Path):
        """Test missing API keys handled gracefully in test environment."""
        env_file = tmp_path / ".env"
        env_file.write_text("")  # No API keys

        os.environ['PYTEST_CURRENT_TEST'] = 'test'
        try:
            config = ConfigurationManager(env_file=env_file)
            # Should not raise in test environment
            config.validate()
        finally:
            if 'PYTEST_CURRENT_TEST' in os.environ:
                del os.environ['PYTEST_CURRENT_TEST']


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_configuration(self, test_config):
        """Test valid configuration passes validation."""
        config = test_config

        # Should not raise
        config.validate()

    def test_directory_creation(self, tmp_path: Path):
        """Test configuration creates required directories."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DEEPGRAM_API_KEY=test-key-12345678\n"
            f"TEMP_DIR={tmp_path}/temp\n"
            f"OUTPUT_DIR={tmp_path}/output\n"
        )

        os.environ['PYTEST_CURRENT_TEST'] = 'test'
        try:
            config = ConfigurationManager(env_file=env_file)
            _ = config.security

            # Directories should be created
            assert (tmp_path / "temp").exists()
            assert (tmp_path / "output").exists()
        finally:
            if 'PYTEST_CURRENT_TEST' in os.environ:
                del os.environ['PYTEST_CURRENT_TEST']

    def test_error_messages_actionable(self, tmp_path: Path):
        """Verify error messages provide clear guidance."""
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPGRAM_API_KEY=bad\n")  # Too short

        os.environ['PYTEST_CURRENT_TEST'] = 'test'
        try:
            config = ConfigurationManager(env_file=env_file)
            with pytest.raises(Exception) as exc_info:
                _ = config.providers

            # Error message should mention .env.example or provide guidance
            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ['invalid', 'short', 'api', 'key'])
        finally:
            if 'PYTEST_CURRENT_TEST' in os.environ:
                del os.environ['PYTEST_CURRENT_TEST']


class TestSecurityPatternDetection:
    """Test detection of insecure patterns in codebase."""

    def test_no_hardcoded_api_keys_in_tests(self):
        """Verify no hardcoded API keys in test files (beyond test fixtures)."""
        import ast
        import re

        # Pattern for API key-like strings (not test keys)
        # Real keys typically: provider-hash or sk-hash format
        api_key_pattern = re.compile(
            r'(?:deepgram|elevenlabs|sk)-[a-f0-9]{20,}',
            re.IGNORECASE
        )

        violations = []
        test_dir = Path(__file__).parent

        for test_file in test_dir.rglob("*.py"):
            if test_file.name in ['api_mocks.py', 'conftest.py', 'test_security.py']:
                # Skip files with intentional test keys
                continue

            content = test_file.read_text()

            # Check for API key patterns
            matches = api_key_pattern.findall(content)
            # Filter out test keys
            real_keys = [m for m in matches if 'test-key' not in m.lower()]

            if real_keys:
                violations.append(f"{test_file.name}: {real_keys}")

        assert not violations, f"Found potential hardcoded API keys: {violations}"
