"""Tests for OpenAI Whisper transcription provider."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.transcription import TranscriptionResult
from src.providers.whisper import WhisperTranscriber


class TestWhisperTranscriber:
    """Test Whisper transcription provider functionality."""

    @pytest.fixture
    def whisper_transcriber(self):
        """Create a WhisperTranscriber instance for testing."""
        return WhisperTranscriber()

    def test_validate_configuration_with_dependencies(self, whisper_transcriber):
        """Test configuration validation when dependencies are available."""
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            with patch("torch.cuda.is_available", return_value=True):
                assert whisper_transcriber.validate_configuration() is True

    def test_validate_configuration_without_dependencies(self, whisper_transcriber):
        """Test configuration validation when dependencies are missing."""
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", False):
            assert whisper_transcriber.validate_configuration() is False

    def test_get_provider_name(self, whisper_transcriber):
        """Test getting provider name."""
        whisper_transcriber.model_name = "base"
        assert whisper_transcriber.get_provider_name() == "OpenAI Whisper (base)"

    def test_get_supported_features(self, whisper_transcriber):
        """Test getting supported features."""
        features = whisper_transcriber.get_supported_features()
        assert "timestamps" in features
        assert "word_timestamps" in features
        assert "language_detection" in features
        assert "local_processing" in features
        assert "offline_capable" in features

    @pytest.mark.asyncio
    async def test_load_model_success(self, whisper_transcriber):
        """Test successful model loading."""
        mock_model = Mock()
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            # Mock the whisper module to avoid import issues
            with patch("src.providers.whisper.whisper") as mock_whisper:
                mock_whisper.load_model.return_value = mock_model
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_model)

                    await whisper_transcriber._load_model()
                    assert whisper_transcriber.model == mock_model

    @pytest.mark.asyncio
    async def test_load_model_failure(self, whisper_transcriber):
        """Test model loading failure."""
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            # Mock the whisper module to avoid import issues
            with patch("src.providers.whisper.whisper") as mock_whisper:
                mock_whisper.load_model.return_value = Mock()
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=Exception("Load failed")
                    )

                    with pytest.raises(Exception, match="Load failed"):
                        await whisper_transcriber._load_model()

    @pytest.mark.asyncio
    async def test_transcribe_impl_success(self, whisper_transcriber):
        """Test successful transcription implementation."""
        # Mock dependencies
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "avg_logprob": 0.8,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": "world", "start": 0.5, "end": 1.5, "probability": 0.8},
                    ],
                }
            ],
            "language": "en",
        }

        whisper_transcriber.model = mock_model

        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            # Mock file exists check
            with patch("pathlib.Path.exists", return_value=True):
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        return_value=mock_model.transcribe.return_value
                    )

                    audio_path = Path("/fake/audio.mp3")
                    result = await whisper_transcriber._transcribe_impl(audio_path, "en")

                    assert result is not None
                    assert isinstance(result, TranscriptionResult)
                    assert result.metadata["language"] == "en"
                    assert result.transcript == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_impl_file_not_found(self, whisper_transcriber):
        """Test transcription with non-existent file."""
        audio_path = Path("/nonexistent/audio.mp3")

        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            # Mock file exists check to return False
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(FileNotFoundError):
                    await whisper_transcriber._transcribe_impl(audio_path, "en")

    def test_parse_whisper_result(self, whisper_transcriber):
        """Test parsing Whisper result into TranscriptionResult."""
        whisper_result = {
            "segments": [
                {
                    "text": "Test transcription",
                    "start": 0.0,
                    "end": 2.0,
                    "avg_logprob": 0.85,
                    "words": [
                        {"word": "Test", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": "transcription", "start": 0.5, "end": 2.0, "probability": 0.8},
                    ],
                }
            ],
            "language": "en",
        }

        audio_path = Path("/fake/audio.mp3")
        processing_time = 5.0

        result = whisper_transcriber._parse_whisper_result(
            whisper_result, audio_path, processing_time
        )

        assert isinstance(result, TranscriptionResult)
        assert result.audio_file == str(audio_path)
        assert result.metadata["language"] == "en"
        assert result.metadata["processing_time_seconds"] == 5.0
        assert result.transcript == "Test transcription"

    def test_extract_words(self, whisper_transcriber):
        """Test word extraction from Whisper segment."""
        segment = {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.8},
            ]
        }

        words = whisper_transcriber._extract_words(segment)

        assert words is not None
        assert len(words) == 2
        assert words[0]["word"] == "Hello"
        assert words[1]["word"] == "world"

    def test_extract_words_no_words(self, whisper_transcriber):
        """Test word extraction when no words are present."""
        segment = {"text": "Hello world", "start": 0.0, "end": 1.0}

        words = whisper_transcriber._extract_words(segment)

        assert words is None

    def test_generate_chapters(self, whisper_transcriber):
        """Test chapter generation from utterances."""
        from src.models.transcription import TranscriptionUtterance

        utterances = [
            TranscriptionUtterance(
                speaker=1, start=0.0, end=300.0, text="Chapter 1 content"  # 5 minutes
            ),
            TranscriptionUtterance(
                speaker=1, start=300.0, end=600.0, text="Chapter 2 content"  # 10 minutes
            ),
        ]

        chapters = whisper_transcriber._generate_chapters(utterances)

        assert len(chapters) >= 2  # At least 2 chapters for 10 minutes of content
        assert chapters[0]["start_time"] == 0.0
        assert chapters[0]["end_time"] == 300.0

    @pytest.mark.asyncio
    async def test_health_check_async_success(self, whisper_transcriber):
        """Test successful health check."""
        mock_model = Mock()
        whisper_transcriber.model = mock_model

        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            with patch("torch.cuda.is_available", return_value=True):
                health = await whisper_transcriber.health_check_async()

                assert health["healthy"] is True
                assert health["status"] == "ready"
                assert "model_loaded" in health["details"]

    @pytest.mark.asyncio
    async def test_health_check_async_no_dependencies(self, whisper_transcriber):
        """Test health check without dependencies."""
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", False):
            health = await whisper_transcriber.health_check_async()

            assert health["healthy"] is False
            assert health["status"] == "dependencies_missing"

    def test_get_model_size_info(self, whisper_transcriber):
        """Test getting model size information."""
        whisper_transcriber.model_name = "base"

        size_info = whisper_transcriber.get_model_size_info()

        assert "current_model" in size_info
        assert "current_info" in size_info
        assert "available_models" in size_info
        assert "model_sizes" in size_info
        assert size_info["current_model"] == "base"
        assert "params" in size_info["current_info"]

    @pytest.mark.asyncio
    async def test_transcribe_async_with_circuit_breaker(self, whisper_transcriber):
        """Test transcription with circuit breaker protection."""
        # Mock successful transcription
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
            "language": "en",
        }
        whisper_transcriber.model = mock_model

        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=mock_model.transcribe.return_value
                )

                # Mock file exists check
                with patch("pathlib.Path.exists", return_value=True):
                    audio_path = Path("/fake/audio.mp3")
                    result = await whisper_transcriber.transcribe_async(audio_path, "en")

                    assert result is not None
                    assert result.transcript == "Test"

    @pytest.mark.asyncio
    async def test_transcribe_async_failure(self, whisper_transcriber):
        """Test transcription failure handling."""
        # Mock failed transcription
        with patch("src.providers.whisper.PROVIDER_AVAILABLE", True):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=Exception("Transcription failed")
                )

                # Mock file exists check
                with patch("pathlib.Path.exists", return_value=True):
                    audio_path = Path("/fake/audio.mp3")
                    result = await whisper_transcriber.transcribe_async(audio_path, "en")

                    assert result is None
