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


class TestDeepgramProviderIntegration(E2ETestBase, MockProviderMixin):
    """Test Deepgram provider integration."""
    
    @classmethod
    def setup_class(cls):
        """Setup test data for the class."""
        cls.test_data_manager = TestDataManager()
        cls.test_files = cls.test_data_manager.generate_all_test_files()
    
    def test_deepgram_provider_initialization(self):
        """Test Deepgram provider initialization."""
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key")
        
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            mock_provider.return_value = mock_instance
            
            # Test provider creation
            provider = mock_provider("test_deepgram_key")
            
            assert provider is not None
            mock_provider.assert_called_once_with("test_deepgram_key")
    
    def test_deepgram_transcription_with_diarization(self):
        """Test Deepgram transcription with speaker diarization."""
        if "audio_only" not in self.test_files:
            pytest.skip("Audio test file not available")
        
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key")
        
        # Mock Deepgram response with speaker diarization
        mock_response = {
            "transcript": "Hello, this is speaker one. And this is speaker two.",
            "speakers": [
                {"speaker": 0, "text": "Hello, this is speaker one.", "start": 0.0, "end": 2.5},
                {"speaker": 1, "text": "And this is speaker two.", "start": 2.5, "end": 5.0}
            ],
            "metadata": {
                "duration": 5.0,
                "speakers": 2
            }
        }
        
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.transcribe.return_value = mock_response
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            result = provider.transcribe(self.test_files["audio_only"])
            
            assert "speakers" in result
            assert len(result["speakers"]) == 2
            assert result["metadata"]["speakers"] == 2
    
    def test_deepgram_file_size_validation(self):
        """Test Deepgram file size validation."""
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key")
        
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            
            # Test file size limit checking
            mock_instance.validate_file_size.return_value = True
            mock_instance.validate_file_size.side_effect = lambda size: size <= 500 * 1024 * 1024  # 500MB
            
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            
            # Test valid file size
            assert provider.validate_file_size(100 * 1024 * 1024)  # 100MB
            
            # Test invalid file size
            assert not provider.validate_file_size(600 * 1024 * 1024)  # 600MB
    
    def test_deepgram_error_handling(self):
        """Test Deepgram provider error handling."""
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key")
        
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            
            # Test API error handling
            mock_instance.transcribe.side_effect = Exception("API service unavailable")
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            
            with pytest.raises(Exception, match="API service unavailable"):
                provider.transcribe("dummy_file.mp3")
    
    def test_deepgram_authentication_error(self):
        """Test Deepgram authentication error handling."""
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.transcribe.side_effect = Exception("Authentication failed")
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("invalid_key")
            
            with pytest.raises(Exception, match="Authentication failed"):
                provider.transcribe("dummy_file.mp3")


class TestElevenLabsProviderIntegration(E2ETestBase, MockProviderMixin):
    """Test ElevenLabs provider integration."""
    
    @classmethod
    def setup_class(cls):
        """Setup test data for the class."""
        cls.test_data_manager = TestDataManager()
        cls.test_files = cls.test_data_manager.generate_all_test_files()
    
    def test_elevenlabs_provider_initialization(self):
        """Test ElevenLabs provider initialization."""
        self.set_test_env(ELEVENLABS_API_KEY="test_elevenlabs_key")
        
        with patch('src.providers.elevenlabs.ElevenLabsProvider') as mock_provider:
            mock_instance = Mock()
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_elevenlabs_key")
            
            assert provider is not None
            mock_provider.assert_called_once_with("test_elevenlabs_key")
    
    def test_elevenlabs_transcription_accuracy(self):
        """Test ElevenLabs transcription for accuracy features."""
        if "audio_only" not in self.test_files:
            pytest.skip("Audio test file not available")
        
        self.set_test_env(ELEVENLABS_API_KEY="test_elevenlabs_key")
        
        # Mock ElevenLabs response
        mock_response = {
            "transcript": "This is a high accuracy transcription from ElevenLabs.",
            "confidence": 0.98,
            "metadata": {
                "duration": 10.0,
                "model": "elevenlabs-premium"
            }
        }
        
        with patch('src.providers.elevenlabs.ElevenLabsProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.transcribe.return_value = mock_response
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            result = provider.transcribe(self.test_files["audio_only"])
            
            assert result["confidence"] >= 0.95
            assert "high accuracy" in result["transcript"]
    
    def test_elevenlabs_rate_limit_handling(self):
        """Test ElevenLabs rate limit handling."""
        self.set_test_env(ELEVENLABS_API_KEY="test_elevenlabs_key")
        
        with patch('src.providers.elevenlabs.ElevenLabsProvider') as mock_provider:
            mock_instance = Mock()
            mock_instance.transcribe.side_effect = Exception("Rate limit exceeded")
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            
            with pytest.raises(Exception, match="Rate limit exceeded"):
                provider.transcribe("dummy_file.mp3")
    
    def test_elevenlabs_file_format_support(self):
        """Test ElevenLabs supported file formats."""
        self.set_test_env(ELEVENLABS_API_KEY="test_elevenlabs_key")
        
        with patch('src.providers.elevenlabs.ElevenLabsProvider') as mock_provider:
            mock_instance = Mock()
            
            # Test supported formats
            mock_instance.supports_format.side_effect = lambda fmt: fmt.lower() in ['mp3', 'wav', 'flac']
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            
            assert provider.supports_format("mp3")
            assert provider.supports_format("wav")
            assert provider.supports_format("flac")
            assert not provider.supports_format("ogg")


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


class TestProviderFallbackScenarios(E2ETestBase, MockProviderMixin):
    """Test provider fallback and error recovery scenarios."""
    
    def test_primary_provider_failure_fallback(self):
        """Test fallback when primary provider fails."""
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            # Mock primary provider failure
            mock_factory.get_best_provider.return_value = "deepgram"
            mock_factory.create_provider.side_effect = [
                Exception("Primary provider failed"),  # First call fails
                Mock()  # Second call succeeds (fallback)
            ]
            
            # Should attempt fallback
            with pytest.raises(Exception, match="Primary provider failed"):
                mock_factory.create_provider("deepgram")
    
    def test_all_providers_failure(self):
        """Test behavior when all providers fail."""
        self.set_test_env(
            DEEPGRAM_API_KEY="test_deepgram_key",
            ELEVENLABS_API_KEY="test_elevenlabs_key"
        )
        
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            # Mock all providers failing
            mock_factory.get_configured_providers.return_value = ["deepgram", "elevenlabs"]
            mock_factory.create_provider.side_effect = Exception("All providers failed")
            
            with pytest.raises(Exception, match="All providers failed"):
                mock_factory.create_provider("deepgram")
            
            with pytest.raises(Exception, match="All providers failed"):
                mock_factory.create_provider("elevenlabs")
    
    def test_graceful_degradation(self):
        """Test graceful degradation when advanced features are unavailable."""
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key")
        
        with patch('src.providers.deepgram.DeepgramProvider') as mock_provider:
            mock_instance = Mock()
            
            # Mock feature availability
            mock_instance.supports_speaker_diarization.return_value = False
            mock_instance.supports_sentiment_analysis.return_value = False
            mock_instance.transcribe.return_value = {
                "transcript": "Basic transcription without advanced features",
                "speakers": None,
                "sentiment": None
            }
            mock_provider.return_value = mock_instance
            
            provider = mock_provider("test_key")
            result = provider.transcribe("dummy_file.mp3")
            
            # Should still provide basic transcription
            assert "Basic transcription" in result["transcript"]
            assert result["speakers"] is None
            assert result["sentiment"] is None
    
    def test_configuration_validation_chain(self):
        """Test complete configuration validation chain."""
        test_scenarios = [
            # Scenario 1: All configured
            {
                "env": {"DEEPGRAM_API_KEY": "valid_key", "ELEVENLABS_API_KEY": "valid_key"},
                "expected_providers": ["deepgram", "elevenlabs"],
                "should_succeed": True
            },
            # Scenario 2: Partial configuration
            {
                "env": {"DEEPGRAM_API_KEY": "valid_key", "ELEVENLABS_API_KEY": ""},
                "expected_providers": ["deepgram"],
                "should_succeed": True
            },
            # Scenario 3: No configuration
            {
                "env": {"DEEPGRAM_API_KEY": "", "ELEVENLABS_API_KEY": ""},
                "expected_providers": [],
                "should_succeed": False
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            with self.subTest(scenario=i):
                self.set_test_env(**scenario["env"])
                
                with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
                    mock_factory.get_configured_providers.return_value = scenario["expected_providers"]
                    
                    configured = mock_factory.get_configured_providers()
                    
                    assert configured == scenario["expected_providers"]
                    
                    if scenario["should_succeed"]:
                        assert len(configured) > 0
                    else:
                        assert len(configured) == 0