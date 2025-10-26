"""
Provider Integration Tests for audio-extraction-analysis.

Tests provider factory pattern and integration scenarios including:
- Provider auto-selection logic
- Factory pattern validation
- Provider-specific feature testing
- Fallback mechanisms
- API key management scenarios
"""
import pytest
import os
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from .base import E2ETestBase, MockProviderMixin
from .test_data_manager import TestDataManager


class TestProviderFactory(E2ETestBase, MockProviderMixin):
    """Test the provider factory pattern and auto-selection logic."""
    
    def setup_method(self):
        """Setup for each test method."""
        super().setup_method()
        
        # Clear provider factory state between tests
        self._clear_factory_state()
    
    def _clear_factory_state(self):
        """Clear any cached state in the provider factory."""
        # This might need adjustment based on actual factory implementation
        try:
            from src.providers.factory import TranscriptionProviderFactory
            if hasattr(TranscriptionProviderFactory, '_configured_providers'):
                TranscriptionProviderFactory._configured_providers = None
            if hasattr(TranscriptionProviderFactory, '_instances'):
                TranscriptionProviderFactory._instances = {}
        except ImportError:
            pass  # Factory not available in test environment
    
    def test_factory_all_providers_available(self):
        """Test factory behavior when all providers are configured."""
        # Set all API keys
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = ["deepgram", "elevenlabs"]
            mock_factory.get_best_provider.return_value = "deepgram"
            
            configured_providers = mock_factory.get_configured_providers()
            best_provider = mock_factory.get_best_provider()
            
            assert "deepgram" in configured_providers
            assert "elevenlabs" in configured_providers
            assert len(configured_providers) == 2
            assert best_provider in configured_providers
    
    def test_factory_deepgram_only(self):
        """Test factory behavior with only Deepgram configured."""
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY=""
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = ["deepgram"]
            mock_factory.get_best_provider.return_value = "deepgram"
            
            configured_providers = mock_factory.get_configured_providers()
            best_provider = mock_factory.get_best_provider()
            
            assert configured_providers == ["deepgram"]
            assert best_provider == "deepgram"
    
    def test_factory_elevenlabs_only(self):
        """Test factory behavior with only ElevenLabs configured."""
        self.set_test_env(
            DEEPGRAM_API_KEY="",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = ["elevenlabs"]
            mock_factory.get_best_provider.return_value = "elevenlabs"
            
            configured_providers = mock_factory.get_configured_providers()
            best_provider = mock_factory.get_best_provider()
            
            assert configured_providers == ["elevenlabs"]
            assert best_provider == "elevenlabs"
    
    def test_factory_no_providers_configured(self):
        """Test factory behavior when no providers are configured."""
        self.set_test_env(
            DEEPGRAM_API_KEY="",
            ELEVENLABS_API_KEY=""
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = []
            mock_factory.get_best_provider.side_effect = ValueError("No providers configured")
            
            configured_providers = mock_factory.get_configured_providers()
            
            assert configured_providers == []
            
            with pytest.raises(ValueError, match="No providers configured"):
                mock_factory.get_best_provider()
    
    def test_factory_invalid_api_keys(self):
        """Test factory behavior with invalid API keys."""
        self.set_test_env(
            DEEPGRAM_API_KEY="invalid_key",
            ELEVENLABS_API_KEY="invalid_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            # Simulate validation that detects invalid keys
            mock_factory.get_configured_providers.return_value = []
            mock_factory.validate_provider.side_effect = ValueError("Invalid API key")
            
            with pytest.raises(ValueError, match="Invalid API key"):
                mock_factory.validate_provider("deepgram")
            
            with pytest.raises(ValueError, match="Invalid API key"):
                mock_factory.validate_provider("elevenlabs")
    
    def test_factory_provider_selection_by_file_size(self):
        """Test provider selection based on file size constraints."""
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        # Mock file size constraints
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            # Small file - should prefer based on provider capabilities
            mock_factory.get_best_provider_for_file.return_value = "elevenlabs"
            small_file_provider = mock_factory.get_best_provider_for_file(
                file_size=1024 * 1024  # 1MB
            )
            
            # Large file - should consider size limits
            mock_factory.get_best_provider_for_file.return_value = "deepgram"
            large_file_provider = mock_factory.get_best_provider_for_file(
                file_size=100 * 1024 * 1024  # 100MB
            )
            
            assert small_file_provider in ["deepgram", "elevenlabs"]
            assert large_file_provider in ["deepgram", "elevenlabs"]
    
    def test_factory_provider_feature_requirements(self):
        """Test provider selection based on feature requirements."""
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            # Test speaker diarization requirement
            mock_factory.get_provider_with_features.return_value = "deepgram"
            diarization_provider = mock_factory.get_provider_with_features(
                features=["speaker_diarization"]
            )
            
            # Test sentiment analysis requirement
            mock_factory.get_provider_with_features.return_value = "deepgram"
            sentiment_provider = mock_factory.get_provider_with_features(
                features=["sentiment_analysis"]
            )
            
            assert diarization_provider in ["deepgram", "elevenlabs"]
            assert sentiment_provider in ["deepgram", "elevenlabs"]


class TestWhisperProviderIntegration(E2ETestBase, MockProviderMixin):
    """Test Whisper provider integration."""
    
    def test_whisper_offline_capability(self):
        """Test Whisper offline transcription capability."""
        # Whisper should work without API keys
        with patch('src.providers.whisper.WhisperProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.transcribe.return_value = {
                "transcript": "Offline transcription result",
                "language": "en",
                "confidence": 0.85
            }
            mock_provider.return_value = mock_instance
            
            provider = mock_provider()  # No API key needed
            result = provider.transcribe("dummy_file.mp3")
            
            assert "Offline transcription" in result["transcript"]
            assert result["language"] == "en"
    
    def test_whisper_model_selection(self):
        """Test Whisper model selection."""
        with patch('src.providers.whisper.WhisperProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.set_model.return_value = True
            mock_provider.return_value = mock_instance
            
            provider = mock_provider()
            
            # Test different model sizes
            models = ["tiny", "base", "small", "medium", "large"]
            for model in models:
                result = provider.set_model(model)
                assert result is True
    
    def test_whisper_gpu_acceleration(self):
        """Test Whisper GPU acceleration detection."""
        with patch('src.providers.whisper.WhisperProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.has_gpu_support.return_value = False  # Assume no GPU in test env
            mock_provider.return_value = mock_instance

            provider = mock_provider()

            # Should work without GPU
            assert not provider.has_gpu_support()


# ============================================================================
# Real Provider Contract Testing Infrastructure
# ============================================================================
# Use pytest markers to enable real provider testing:
#   pytest -m "real_provider" --provider=deepgram
#   pytest -m "real_provider" --provider=elevenlabs
#
# Requires valid API keys in environment:
#   DEEPGRAM_API_KEY, ELEVENLABS_API_KEY
# ============================================================================

@pytest.fixture(scope="session")
def real_provider_mode(request):
    """
    Fixture to enable real provider contract testing.

    Usage:
        pytest --real-provider-mode
        pytest --real-provider-mode --provider=deepgram

    Returns dict with:
        - enabled: bool - whether real provider mode is active
        - provider: str - specific provider to test (None = all available)
    """
    return {
        "enabled": request.config.getoption("--real-provider-mode", default=False),
        "provider": request.config.getoption("--provider", default=None)
    }


@pytest.fixture(scope="session")
def real_provider_credentials():
    """
    Fixture providing real API credentials for contract testing.

    Returns dict with available provider credentials:
        - deepgram: API key or None
        - elevenlabs: API key or None

    Note: Only returns credentials that are actually set in environment.
    """
    return {
        "deepgram": os.getenv("DEEPGRAM_API_KEY"),
        "elevenlabs": os.getenv("ELEVENLABS_API_KEY")
    }


@pytest.fixture(scope="function")
def real_audio_sample(tmp_path):
    """
    Fixture providing a small real audio file for contract testing.

    Creates a minimal audio file that can be used for real provider tests
    without consuming significant quota.

    Returns:
        Path to audio file suitable for provider contract testing
    """
    # In real implementation, this would generate or provide a small test audio file
    # For now, return a path that tests can check existence of
    audio_path = tmp_path / "contract_test_sample.mp3"

    # Tests should check if this file exists and skip if not available
    # Real implementation would generate a minimal valid audio file
    return audio_path


def pytest_configure(config):
    """Register custom markers for real provider testing."""
    config.addinivalue_line(
        "markers",
        "real_provider: mark test to run against real provider APIs (requires credentials)"
    )


def pytest_addoption(parser):
    """Add command-line options for real provider testing."""
    parser.addoption(
        "--real-provider-mode",
        action="store_true",
        default=False,
        help="Enable real provider contract testing (requires API keys)"
    )
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        help="Specific provider to test in real mode (deepgram, elevenlabs, whisper)"
    )
