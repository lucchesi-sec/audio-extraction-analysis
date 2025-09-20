import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def clear_deepgram_env(monkeypatch):
    """Ensure DEEPGRAM_API_KEY is absent by default; tests can set it explicitly when needed."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)


@pytest.fixture
def mock_deepgram_client(monkeypatch):
    """Mock the Deepgram client to prevent external API calls during testing."""

    # Mock response data structure matching Deepgram Nova 3 API
    mock_response_data = {
        "metadata": {"duration": 120.5, "channels": 1},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": (
                                "This is a test transcription with multiple speakers "
                                "discussing important topics."
                            ),
                            "paragraphs": {
                                "paragraphs": [
                                    {
                                        "sentences": [
                                            {
                                                "text": (
                                                    "This is a test transcription with "
                                                    "multiple speakers discussing important topics."
                                                )
                                            }
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            ],
            "summary": {"short": "A discussion about important topics between multiple speakers."},
            "topics": {
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 60.0,
                        "topics": [
                            {"topic": "Technology", "confidence_score": 0.85},
                            {"topic": "Business", "confidence_score": 0.72},
                        ],
                    },
                    {
                        "start_time": 60.0,
                        "end_time": 120.5,
                        "topics": [{"topic": "Innovation", "confidence_score": 0.91}],
                    },
                ]
            },
            "intents": {"segments": [{"intents": [{"intent": "inform"}, {"intent": "question"}]}]},
            "sentiments": {
                "segments": [
                    {"sentiment": "positive"},
                    {"sentiment": "neutral"},
                    {"sentiment": "positive"},
                ]
            },
            "utterances": [
                {
                    "speaker": 0,
                    "start": 0.0,
                    "end": 30.0,
                    "transcript": "This is the first speaker talking about technology.",
                },
                {
                    "speaker": 1,
                    "start": 30.0,
                    "end": 90.0,
                    "transcript": "This is the second speaker discussing business innovations.",
                },
                {
                    "speaker": 0,
                    "start": 90.0,
                    "end": 120.5,
                    "transcript": "First speaker concluding the discussion.",
                },
            ],
        },
    }

    # Create mock response object with proper attribute access
    mock_response = Mock()
    mock_response.metadata.duration = mock_response_data["metadata"]["duration"]
    mock_response.results.channels = [
        Mock(
            alternatives=[
                Mock(
                    transcript=mock_response_data["results"]["channels"][0]["alternatives"][0][
                        "transcript"
                    ],
                    paragraphs=Mock(
                        paragraphs=[
                            Mock(
                                sentences=[
                                    Mock(
                                        text=(
                                            "This is a test transcription with multiple speakers "
                                            "discussing important topics."
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                )
            ]
        )
    ]

    # Mock summary
    mock_response.results.summary = Mock()
    mock_response.results.summary.short = mock_response_data["results"]["summary"]["short"]

    # Mock topics
    mock_topics = Mock()
    mock_topics.segments = [
        Mock(
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            topics=[
                Mock(topic=t["topic"], confidence_score=t["confidence_score"])
                for t in seg["topics"]
            ],
        )
        for seg in mock_response_data["results"]["topics"]["segments"]
    ]
    mock_response.results.topics = mock_topics

    # Mock intents
    mock_intents = Mock()
    mock_intents.segments = [
        Mock(intents=[Mock(intent=i["intent"]) for i in seg["intents"]])
        for seg in mock_response_data["results"]["intents"]["segments"]
    ]
    mock_response.results.intents = mock_intents

    # Mock sentiments
    mock_sentiments = Mock()
    mock_sentiments.segments = [
        Mock(sentiment=seg["sentiment"])
        for seg in mock_response_data["results"]["sentiments"]["segments"]
    ]
    mock_response.results.sentiments = mock_sentiments

    # Mock utterances
    mock_utterances = [
        Mock(
            speaker=utt["speaker"], start=utt["start"], end=utt["end"], transcript=utt["transcript"]
        )
        for utt in mock_response_data["results"]["utterances"]
    ]
    mock_response.results.utterances = mock_utterances

    # Mock the Deepgram client
    mock_client = Mock()
    mock_client.listen.asyncrest.v.return_value.transcribe_file.return_value = mock_response

    # Mock the Deepgram SDK imports
    mock_deepgram = Mock()
    mock_deepgram.ClientOptionsFromEnv = Mock(return_value=Mock())
    mock_deepgram.DeepgramClient = Mock(return_value=mock_client)
    mock_deepgram.PrerecordedOptions = Mock()

    # Patch the lazy import in transcription service
    def mock_import():
        return (
            mock_deepgram.ClientOptionsFromEnv,
            mock_deepgram.DeepgramClient,
            mock_deepgram.PrerecordedOptions,
        )

    # We'll patch this in individual tests since it's a local import
    return mock_deepgram, mock_response


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        # Write some dummy audio data (just bytes, not real audio)
        temp_file.write(b"fake_audio_data" * 1000)
        temp_path = Path(temp_file.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        # Write some dummy video data
        temp_file.write(b"fake_video_data" * 2000)
        temp_path = Path(temp_file.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_video_file_with_brackets():
    """Create a temporary video file with square brackets in the name for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with square brackets in the name
        video_file = Path(temp_dir) / "test[with_square_brackets].mp4"
        video_file.write_bytes(b"fake_video_data" * 2000)
        yield video_file


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_ffmpeg(monkeypatch):
    """Mock FFmpeg subprocess calls to prevent actual video processing."""
    mock_run = Mock()
    mock_run.return_value = Mock(returncode=0, stderr="", stdout="Duration: 00:02:00.50")

    monkeypatch.setattr("subprocess.run", mock_run)

    return mock_run


@pytest.fixture
def api_key_set(monkeypatch):
    """Set a mock API key for tests that need it."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test_api_key_12345")
