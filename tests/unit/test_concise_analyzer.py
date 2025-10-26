"""Tests for the ConciseAnalyzer that generates a single comprehensive analysis file."""

from datetime import datetime
from pathlib import Path

from src.analysis.concise_analyzer import ConciseAnalyzer
from src.models.transcription import (
    TranscriptionChapter,
    TranscriptionResult,
    TranscriptionSpeaker,
    TranscriptionUtterance,
)


def _make_basic_result() -> TranscriptionResult:
    """Create a basic transcription result for testing."""
    return TranscriptionResult(
        transcript=(
            "This is a sample transcript. We should follow up on action items. "
            "The team will deliver next week. Planning is a priority. "
            "We need to ensure quality standards are met in the final deliverable. "
            "The deadline is approaching fast and we must coordinate effectively."
        ),
        duration=120.0,
        generated_at=datetime.now(),
        audio_file="/tmp/test_audio.mp3",
        provider_name="Deepgram Nova 3",
        provider_features=["timestamps", "speaker_diarization", "topic_detection"],
    )


def test_concise_analyzer_creates_analysis_file(tmp_path):
    """Test that analyze_and_save creates the analysis file."""
    result = _make_basic_result()
    analyzer = ConciseAnalyzer()

    output_path = analyzer.analyze_and_save(result, tmp_path, "test_video")

    assert output_path.exists()
    assert output_path.name == "test_video_analysis.md"
    assert output_path.parent == tmp_path


def test_concise_analyzer_file_has_content(tmp_path):
    """Test that the generated analysis file has meaningful content."""
    result = _make_basic_result()
    result.summary = "This is a test summary of the audio content."

    analyzer = ConciseAnalyzer()
    output_path = analyzer.analyze_and_save(result, tmp_path, "test")

    content = output_path.read_text(encoding="utf-8")

    # Verify all major sections are present
    assert "# Audio Analysis Report" in content
    assert "## 📋 Overview" in content
    assert "## 🎯 Key Topics" in content
    assert "## 👥 Speaker Analysis" in content
    assert "## 😊 Sentiment Analysis" in content
    assert "## 💡 Key Highlights & Quotes" in content
    assert "## ✅ Intents & Action Items" in content
    assert "## ⏰ Timeline" in content
    assert "## 📊 Technical Metadata" in content


def test_header_generation():
    """Test that header is generated correctly with metadata."""
    result = _make_basic_result()
    analyzer = ConciseAnalyzer()

    header = analyzer._generate_header(result)

    assert "# Audio Analysis Report" in header
    assert "Deepgram Nova 3" in header
    assert "02:00" in header  # Duration formatted
    assert "test_audio.mp3" in header


def test_overview_with_summary():
    """Test overview section when summary is provided."""
    result = _make_basic_result()
    result.summary = "This is a comprehensive summary of the content."

    analyzer = ConciseAnalyzer()
    overview = analyzer._generate_overview(result)

    assert "## 📋 Overview" in overview
    assert "This is a comprehensive summary of the content." in overview
    assert "words" in overview


def test_overview_without_summary():
    """Test overview section generates fallback when no summary provided."""
    result = _make_basic_result()
    result.summary = None

    analyzer = ConciseAnalyzer()
    overview = analyzer._generate_overview(result)

    assert "## 📋 Overview" in overview
    assert "Key Content:" in overview


def test_key_topics_with_data():
    """Test key topics section with topic data."""
    result = _make_basic_result()
    result.topics = {
        "planning": 5,
        "execution": 3,
        "quality": 7,
        "deadline": 2,
    }

    analyzer = ConciseAnalyzer()
    topics_section = analyzer._generate_key_topics(result)

    assert "## 🎯 Key Topics" in topics_section
    assert "Quality" in topics_section  # Should be title-cased
    assert "(7 mentions)" in topics_section
    assert "Planning" in topics_section
    assert "(5 mentions)" in topics_section


def test_key_topics_without_data():
    """Test key topics section when no topics are detected."""
    result = _make_basic_result()
    result.topics = {}

    analyzer = ConciseAnalyzer()
    topics_section = analyzer._generate_key_topics(result)

    assert "## 🎯 Key Topics" in topics_section
    assert "No specific topics identified" in topics_section


def test_speaker_insights_with_speakers():
    """Test speaker insights section with speaker data."""
    result = _make_basic_result()
    result.speakers = [
        TranscriptionSpeaker(id=0, total_time=70.0, percentage=58.3),
        TranscriptionSpeaker(id=1, total_time=50.0, percentage=41.7),
    ]

    analyzer = ConciseAnalyzer()
    speaker_section = analyzer._generate_speaker_insights(result)

    assert "## 👥 Speaker Analysis" in speaker_section
    assert "Speaker 0" in speaker_section
    assert "01:10" in speaker_section  # 70 seconds formatted
    assert "58.3%" in speaker_section
    assert "Speaker 1" in speaker_section


def test_speaker_insights_without_speakers():
    """Test speaker insights when no speaker data available."""
    result = _make_basic_result()
    result.speakers = []

    analyzer = ConciseAnalyzer()
    speaker_section = analyzer._generate_speaker_insights(result)

    assert "## 👥 Speaker Analysis" in speaker_section
    assert "No speaker diarization data available" in speaker_section


def test_sentiment_analysis_with_data():
    """Test sentiment analysis section with sentiment data."""
    result = _make_basic_result()
    result.sentiment_distribution = {
        "positive": 10,
        "neutral": 5,
        "negative": 2,
    }

    analyzer = ConciseAnalyzer()
    sentiment_section = analyzer._generate_sentiment_analysis(result)

    assert "## 😊 Sentiment Analysis" in sentiment_section
    assert "😊" in sentiment_section  # Positive emoji
    assert "Positive" in sentiment_section
    assert "10 segments" in sentiment_section
    assert "😐" in sentiment_section  # Neutral emoji
    assert "😔" in sentiment_section  # Negative emoji


def test_sentiment_analysis_without_data():
    """Test sentiment analysis when no data available."""
    result = _make_basic_result()
    result.sentiment_distribution = {}

    analyzer = ConciseAnalyzer()
    sentiment_section = analyzer._generate_sentiment_analysis(result)

    assert "## 😊 Sentiment Analysis" in sentiment_section
    assert "No sentiment analysis data available" in sentiment_section


def test_highlights_and_quotes_extraction():
    """Test that highlights extract meaningful sentences."""
    # Create a result with longer, more meaningful sentences
    result = TranscriptionResult(
        transcript=(
            "This is an introductory statement. "
            "We need to carefully evaluate all the options available before making a final decision on this critical project. "
            "The team has been working diligently on implementing the new features and ensuring they meet all quality standards. "
            "Our primary objective is to deliver exceptional value to customers while maintaining operational efficiency and cost effectiveness. "
            "The stakeholders have expressed strong support for this initiative and we are confident in achieving the desired outcomes."
        ),
        duration=120.0,
        generated_at=datetime.now(),
        audio_file="/tmp/test.mp3",
        provider_name="Test",
    )

    analyzer = ConciseAnalyzer()
    highlights = analyzer._generate_highlights_and_quotes(result)

    assert "## 💡 Key Highlights & Quotes" in highlights
    # Should extract meaningful long sentences
    assert "quality standards" in highlights or "operational efficiency" in highlights or "stakeholders" in highlights


def test_action_items_with_intents():
    """Test action items section when intents are provided."""
    result = _make_basic_result()
    result.intents = ["inform", "decide", "action", "decide"]

    analyzer = ConciseAnalyzer()
    action_section = analyzer._generate_action_items_and_intents(result)

    assert "## ✅ Intents & Action Items" in action_section
    assert "Identified Intents:" in action_section
    assert "Inform" in action_section or "inform" in action_section.lower()
    assert "Decide" in action_section or "decide" in action_section.lower()


def test_action_items_without_intents():
    """Test action items extraction from transcript."""
    result = _make_basic_result()
    result.intents = []

    analyzer = ConciseAnalyzer()
    action_section = analyzer._generate_action_items_and_intents(result)

    assert "## ✅ Intents & Action Items" in action_section
    # Should detect action keywords like "should", "need to", "must"
    assert "should follow up" in action_section or "Potential Action Items" in action_section


def test_timeline_with_chapters():
    """Test timeline section with chapter data."""
    result = _make_basic_result()
    result.chapters = [
        TranscriptionChapter(
            start_time=0.0,
            end_time=60.0,
            topics=["introduction", "planning"],
            confidence_scores=[0.9, 0.7],
        ),
        TranscriptionChapter(
            start_time=60.0,
            end_time=120.0,
            topics=["execution"],
            confidence_scores=[0.8],
        ),
    ]

    analyzer = ConciseAnalyzer()
    timeline = analyzer._generate_timeline(result)

    assert "## ⏰ Timeline" in timeline
    assert "Topic Segments:" in timeline
    assert "00:00" in timeline
    assert "01:00" in timeline
    assert "introduction, planning" in timeline or "introduction" in timeline


def test_timeline_with_utterances():
    """Test timeline section with utterance data when no chapters."""
    result = _make_basic_result()
    result.chapters = []
    result.utterances = [
        TranscriptionUtterance(speaker=0, start=5.0, end=15.0, text="This is the opening statement."),
        TranscriptionUtterance(speaker=1, start=20.0, end=35.0, text="I agree with that approach."),
    ]

    analyzer = ConciseAnalyzer()
    timeline = analyzer._generate_timeline(result)

    assert "## ⏰ Timeline" in timeline
    assert "Key Moments:" in timeline
    assert "00:05" in timeline
    assert "This is the opening statement" in timeline


def test_timeline_without_data():
    """Test timeline when no chapter or utterance data available."""
    result = _make_basic_result()
    result.chapters = []
    result.utterances = []

    analyzer = ConciseAnalyzer()
    timeline = analyzer._generate_timeline(result)

    assert "## ⏰ Timeline" in timeline
    assert "No timeline data available" in timeline


def test_metadata_generation():
    """Test metadata section generation."""
    result = _make_basic_result()
    result.speakers = [TranscriptionSpeaker(id=0, total_time=60.0, percentage=50.0)]
    result.chapters = [
        TranscriptionChapter(start_time=0.0, end_time=60.0, topics=["test"], confidence_scores=[0.8])
    ]
    result.utterances = [
        TranscriptionUtterance(speaker=0, start=0.0, end=10.0, text="Test utterance")
    ]

    analyzer = ConciseAnalyzer()
    metadata = analyzer._generate_metadata(result)

    assert "## 📊 Technical Metadata" in metadata
    assert "timestamps, speaker_diarization, topic_detection" in metadata
    assert "test_audio.mp3" in metadata
    assert "Speakers Detected:** 1" in metadata
    assert "Topic Segments:** 1" in metadata
    assert "Total Utterances:** 1" in metadata


def test_format_duration_seconds_only():
    """Test duration formatting for times under a minute."""
    analyzer = ConciseAnalyzer()

    assert analyzer._format_duration(0) == "00:00"
    assert analyzer._format_duration(30) == "00:30"
    assert analyzer._format_duration(59) == "00:59"


def test_format_duration_minutes():
    """Test duration formatting for times with minutes."""
    analyzer = ConciseAnalyzer()

    assert analyzer._format_duration(60) == "01:00"
    assert analyzer._format_duration(90) == "01:30"
    assert analyzer._format_duration(3599) == "59:59"


def test_format_duration_hours():
    """Test duration formatting for times with hours."""
    analyzer = ConciseAnalyzer()

    assert analyzer._format_duration(3600) == "01:00:00"
    assert analyzer._format_duration(3661) == "01:01:01"
    assert analyzer._format_duration(7200) == "02:00:00"
    assert analyzer._format_duration(7325) == "02:02:05"


def test_get_sentiment_emoji():
    """Test sentiment emoji mapping."""
    analyzer = ConciseAnalyzer()

    assert analyzer._get_sentiment_emoji("positive") == "😊"
    assert analyzer._get_sentiment_emoji("Positive") == "😊"
    assert analyzer._get_sentiment_emoji("negative") == "😔"
    assert analyzer._get_sentiment_emoji("Negative") == "😔"
    assert analyzer._get_sentiment_emoji("neutral") == "😐"
    assert analyzer._get_sentiment_emoji("Neutral") == "😐"
    assert analyzer._get_sentiment_emoji("unknown") == "🤔"


def test_complete_analysis_with_all_features(tmp_path):
    """Integration test with all features enabled."""
    result = _make_basic_result()
    result.summary = "Comprehensive discussion about project planning and execution."
    result.topics = {"planning": 5, "execution": 3, "quality": 2}
    result.speakers = [
        TranscriptionSpeaker(id=0, total_time=70.0, percentage=58.3),
        TranscriptionSpeaker(id=1, total_time=50.0, percentage=41.7),
    ]
    result.chapters = [
        TranscriptionChapter(
            start_time=0.0, end_time=60.0, topics=["planning"], confidence_scores=[0.9]
        ),
        TranscriptionChapter(
            start_time=60.0, end_time=120.0, topics=["execution"], confidence_scores=[0.8]
        ),
    ]
    result.utterances = [
        TranscriptionUtterance(speaker=0, start=5.0, end=15.0, text="Let's discuss the plan."),
        TranscriptionUtterance(speaker=1, start=20.0, end=30.0, text="I agree with the approach."),
    ]
    result.intents = ["inform", "decide", "action"]
    result.sentiment_distribution = {"positive": 8, "neutral": 3, "negative": 1}

    analyzer = ConciseAnalyzer()
    output_path = analyzer.analyze_and_save(result, tmp_path, "complete_test")

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")

    # Verify all sections have rich content
    assert "Comprehensive discussion" in content
    assert "Planning" in content
    assert "Speaker 0" in content
    assert "😊" in content  # Sentiment emoji
    assert "01:00" in content  # Timeline
    assert "inform" in content.lower()  # Intents


def test_analysis_handles_missing_optionals(tmp_path):
    """Test that analyzer gracefully handles missing optional data."""
    result = TranscriptionResult(
        transcript="Simple transcript without advanced features.",
        duration=30.0,
        generated_at=datetime.now(),
        audio_file="/tmp/simple.mp3",
        provider_name="Basic Provider",
    )

    analyzer = ConciseAnalyzer()
    output_path = analyzer.analyze_and_save(result, tmp_path, "minimal")

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")

    # Should still have all sections with fallback text
    assert "# Audio Analysis Report" in content
    assert "No specific topics identified" in content
    assert "No speaker diarization data available" in content
    assert "No sentiment analysis data available" in content
    assert "No timeline data available" in content


def test_output_directory_creation(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    result = _make_basic_result()
    analyzer = ConciseAnalyzer()

    nested_dir = tmp_path / "output" / "analysis" / "reports"
    output_path = analyzer.analyze_and_save(result, nested_dir, "test")

    assert nested_dir.exists()
    assert output_path.exists()
    assert output_path.parent == nested_dir
