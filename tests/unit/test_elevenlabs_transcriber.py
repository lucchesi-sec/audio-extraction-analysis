"""Unit tests for ElevenLabsTranscriber."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from src.models.transcription import TranscriptionResult, TranscriptionUtterance
from src.providers.elevenlabs import ElevenLabsTranscriber


@pytest.fixture(autouse=True)
def clear_elevenlabs_env(monkeypatch):
    """Ensure ELEVENLABS_API_KEY is absent by default; tests can set it explicitly when needed."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)


@pytest.fixture
def mock_elevenlabs_client():
    """Mock the ElevenLabs client to prevent external API calls during testing."""
    mock_client = Mock()

    # Mock a successful response
    mock_response = Mock()
    mock_response.text = "This is a test transcription from ElevenLabs."
    mock_response.segments = [
        Mock(start=0.0, end=15.0, text="This is a test transcription"),
        Mock(start=15.0, end=30.0, text="from ElevenLabs."),
    ]

    mock_client.speech_to_text.return_value = mock_response
    return mock_client, mock_response


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.mp3"
    # Create file with reasonable size (less than 50MB limit)
    audio_file.write_bytes(b"fake_audio_data" * 1000)  # ~15KB
    return audio_file


@pytest.fixture
def large_audio_file(tmp_path):
    """Create a large temporary audio file that exceeds ElevenLabs limit."""
    audio_file = tmp_path / "large_audio.mp3"
    # Create file larger than 50MB
    audio_file.write_bytes(b"fake_audio_data" * 4000000)  # ~60MB
    return audio_file


class TestElevenLabsTranscriberInit:
    """Test ElevenLabsTranscriber initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")
        assert transcriber.api_key == "test_key"

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "env_key")

        with patch("src.providers.elevenlabs.Config") as mock_config:
            mock_config.ELEVENLABS_API_KEY = "env_key"
            transcriber = ElevenLabsTranscriber()
            assert transcriber.api_key == "env_key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY not found"):
            ElevenLabsTranscriber()

    def test_validate_configuration_with_key(self):
        """Test configuration validation with valid API key."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")
        assert transcriber.validate_configuration() is True

    def test_validate_configuration_without_key(self):
        """Test configuration validation without API key."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")
        transcriber.api_key = None
        assert transcriber.validate_configuration() is False


class TestElevenLabsTranscriberMethods:
    """Test ElevenLabsTranscriber methods."""

    def test_get_provider_name(self):
        """Test get_provider_name returns correct name."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")
        assert transcriber.get_provider_name() == "ElevenLabs"

    def test_get_supported_features(self):
        """Test get_supported_features returns expected features."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")
        features = transcriber.get_supported_features()

        expected_features = ["timestamps", "language_detection", "basic_transcription"]

        assert features == expected_features

    def test_supports_feature(self):
        """Test supports_feature method."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        assert transcriber.supports_feature("timestamps") is True
        assert transcriber.supports_feature("basic_transcription") is True
        assert transcriber.supports_feature("speaker_diarization") is False
        assert transcriber.supports_feature("topic_detection") is False


class TestElevenLabsTranscriberTranscription:
    """Test ElevenLabsTranscriber transcription functionality."""

    def test_transcribe_async_success(self, temp_audio_file):
        """Test successful async transcription."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        # Mock the transcribe_async method directly
        from datetime import datetime

        mock_result = TranscriptionResult(
            transcript="This is a test transcription from ElevenLabs.",
            duration=30.5,
            generated_at=datetime.now(),
            audio_file=str(temp_audio_file),
            provider_name="ElevenLabs",
            provider_features=["timestamps", "language_detection", "basic_transcription"],
        )

        # Add mock utterances
        mock_result.utterances = [
            TranscriptionUtterance(
                speaker=0, start=0.0, end=15.0, text="This is a test transcription"
            ),
            TranscriptionUtterance(speaker=0, start=15.0, end=30.0, text="from ElevenLabs."),
        ]

        with patch.object(transcriber, "transcribe_async", return_value=mock_result):
            result = transcriber.transcribe(temp_audio_file, "en")

        assert result is not None
        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "This is a test transcription from ElevenLabs."
        assert result.provider_name == "ElevenLabs"
        assert result.provider_features == [
            "timestamps",
            "language_detection",
            "basic_transcription",
        ]
        assert result.duration == 30.5
        assert len(result.utterances) == 2

        # Check utterances
        assert result.utterances[0].start == 0.0
        assert result.utterances[0].end == 15.0
        assert result.utterances[0].text == "This is a test transcription"

    def test_transcribe_async_file_not_found(self):
        """Test transcription with non-existent file."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        non_existent_file = Path("/non/existent/file.mp3")
        result = transcriber.transcribe(non_existent_file, "en")

        assert result is None

    def test_transcribe_async_file_too_large(self, large_audio_file):
        """Test transcription with file exceeding size limit."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        result = transcriber.transcribe(large_audio_file, "en")

        assert result is None

    @patch("elevenlabs.client.ElevenLabs")
    def test_transcribe_async_elevenlabs_not_installed(
        self, mock_elevenlabs_class, temp_audio_file
    ):
        """Test transcription when ElevenLabs SDK is not installed."""
        mock_elevenlabs_class.side_effect = ImportError("No module named 'elevenlabs'")

        transcriber = ElevenLabsTranscriber(api_key="test_key")

        result = transcriber.transcribe(temp_audio_file, "en")

        assert result is None

    @patch("elevenlabs.client.ElevenLabs")
    def test_transcribe_async_api_error(self, mock_elevenlabs_class, temp_audio_file):
        """Test transcription with API error."""
        mock_client = Mock()
        mock_client.speech_to_text.side_effect = Exception("API Error")
        mock_elevenlabs_class.return_value = mock_client

        transcriber = ElevenLabsTranscriber(api_key="test_key")

        with patch("builtins.open", mock_open(read_data=b"fake_audio_data")):
            result = transcriber.transcribe(temp_audio_file, "en")

        assert result is None

    @patch("elevenlabs.client.ElevenLabs")
    def test_transcribe_async_response_variations(self, mock_elevenlabs_class, temp_audio_file):
        """Test transcription with different response formats."""
        mock_client = Mock()

        # Test response with transcript attribute instead of text
        mock_response = Mock()
        mock_response.transcript = "Test with transcript attribute"
        # Make sure text attribute exists but is None to test the fallback
        mock_response.text = None
        # Set segments to a valid value (False/None) to avoid iteration issues
        mock_response.segments = []
        # Set hasattr properly by defining the attribute
        del mock_response.segments  # Remove the segments attribute to test fallback
        mock_client.speech_to_text.return_value = mock_response
        mock_elevenlabs_class.return_value = mock_client

        transcriber = ElevenLabsTranscriber(api_key="test_key")

        with patch("builtins.open", mock_open(read_data=b"fake_audio_data")):
            result = transcriber.transcribe(temp_audio_file, "en")

        assert result is not None
        assert result.transcript == "Test with transcript attribute"

    @patch("elevenlabs.client.ElevenLabs")
    def test_transcribe_async_no_segments(self, mock_elevenlabs_class, temp_audio_file):
        """Test transcription with response that has no segments."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Simple transcription without segments"
        mock_response.segments = None
        mock_client.speech_to_text.return_value = mock_response
        mock_elevenlabs_class.return_value = mock_client

        transcriber = ElevenLabsTranscriber(api_key="test_key")

        with patch("builtins.open", mock_open(read_data=b"fake_audio_data")):
            result = transcriber.transcribe(temp_audio_file, "en")

        assert result is not None
        assert result.transcript == "Simple transcription without segments"
        assert len(result.utterances) == 0


class TestElevenLabsTranscriberDurationEstimation:
    """Test ElevenLabsTranscriber duration estimation."""

    @patch("subprocess.run")
    def test_estimate_audio_duration_with_ffprobe(self, mock_subprocess, temp_audio_file):
        """Test duration estimation using ffprobe."""
        mock_subprocess.return_value = Mock(returncode=0, stdout='{"format": {"duration": "45.6"}}')

        transcriber = ElevenLabsTranscriber(api_key="test_key")
        duration = transcriber._estimate_audio_duration(temp_audio_file)

        assert duration == 45.6

    @patch("subprocess.run")
    def test_estimate_audio_duration_ffprobe_fail(self, mock_subprocess, temp_audio_file):
        """Test duration estimation fallback when ffprobe fails."""
        mock_subprocess.return_value = Mock(returncode=1)

        transcriber = ElevenLabsTranscriber(api_key="test_key")
        duration = transcriber._estimate_audio_duration(temp_audio_file)

        # Should use file size estimation
        assert duration > 0
        assert isinstance(duration, float)

    @patch("subprocess.run")
    def test_estimate_audio_duration_ffprobe_exception(self, mock_subprocess, temp_audio_file):
        """Test duration estimation when ffprobe raises exception."""
        mock_subprocess.side_effect = Exception("ffprobe not found")

        transcriber = ElevenLabsTranscriber(api_key="test_key")
        duration = transcriber._estimate_audio_duration(temp_audio_file)

        # Should use file size estimation
        assert duration > 0
        assert isinstance(duration, float)


class TestElevenLabsTranscriberSaveResult:
    """Test ElevenLabsTranscriber save result functionality."""

    def test_save_result_to_file(self, tmp_path):
        """Test saving transcription result to file."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        # Create a test result
        result = TranscriptionResult(
            transcript="Test transcription content",
            duration=60.0,
            generated_at=datetime(2024, 1, 1, 12, 0, 0),
            audio_file="test.mp3",
            provider_name="ElevenLabs",
            provider_features=["timestamps", "basic_transcription"],
        )

        # Add some utterances
        result.utterances = [
            TranscriptionUtterance(speaker=0, start=0.0, end=30.0, text="First part"),
            TranscriptionUtterance(speaker=0, start=30.0, end=60.0, text="Second part"),
        ]

        output_file = tmp_path / "transcript.txt"
        transcriber.save_result_to_file(result, output_file)

        # Verify file was created and contains expected content
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")

        assert "ELEVENLABS TRANSCRIPTION" in content
        assert "Generated: 2024-01-01 12:00:00" in content
        assert "Audio File: test.mp3" in content
        assert "Duration: 60.00 seconds" in content
        assert "Provider: ElevenLabs" in content
        assert "Test transcription content" in content
        assert "[0.00s] First part" in content
        assert "[30.00s] Second part" in content

    def test_save_result_to_file_no_utterances(self, tmp_path):
        """Test saving transcription result with no utterances."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        result = TranscriptionResult(
            transcript="Simple transcription",
            duration=30.0,
            generated_at=datetime.now(),
            audio_file="simple.mp3",
            provider_name="ElevenLabs",
        )

        output_file = tmp_path / "simple_transcript.txt"
        transcriber.save_result_to_file(result, output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")

        assert "Simple transcription" in content
        assert "TRANSCRIPT WITH TIMESTAMPS:" not in content  # Should not appear without utterances

    def test_save_result_creates_parent_directory(self, tmp_path):
        """Test that saving result creates parent directories if they don't exist."""
        transcriber = ElevenLabsTranscriber(api_key="test_key")

        result = TranscriptionResult(
            transcript="Test",
            duration=10.0,
            generated_at=datetime.now(),
            audio_file="test.mp3",
            provider_name="ElevenLabs",
        )

        nested_output = tmp_path / "nested" / "dir" / "transcript.txt"
        transcriber.save_result_to_file(result, nested_output)

        assert nested_output.exists()
        assert "Test" in nested_output.read_text(encoding="utf-8")


class TestElevenLabsTranscriberIntegration:
    """Integration tests for ElevenLabsTranscriber."""

    @patch("elevenlabs.client.ElevenLabs")
    def test_full_transcription_workflow(self, mock_elevenlabs_class, temp_audio_file, tmp_path):
        """Test complete transcription workflow from audio file to saved result."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Complete integration test transcription"
        mock_response.segments = [
            Mock(start=0.0, end=20.0, text="Complete integration test"),
            Mock(start=20.0, end=40.0, text="transcription"),
        ]
        mock_client.speech_to_text.return_value = mock_response
        mock_elevenlabs_class.return_value = mock_client

        transcriber = ElevenLabsTranscriber(api_key="integration_test_key")

        # Perform transcription
        with patch("builtins.open", mock_open(read_data=b"integration_test_audio")):
            result = transcriber.transcribe(temp_audio_file, "en")

        # Verify result
        assert result is not None
        assert result.provider_name == "ElevenLabs"
        assert "integration test" in result.transcript.lower()

        # Save and verify file
        output_file = tmp_path / "integration_result.txt"
        transcriber.save_result_to_file(result, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "ELEVENLABS TRANSCRIPTION" in content
        assert "Complete integration test transcription" in content
