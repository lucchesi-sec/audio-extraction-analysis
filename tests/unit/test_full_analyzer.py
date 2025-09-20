"""Tests for the FullAnalyzer that generates 5 markdown files."""

from datetime import datetime

from src.analysis.full_analyzer import FullAnalyzer
from src.models.transcription import (
    TranscriptionChapter,
    TranscriptionResult,
    TranscriptionSpeaker,
    TranscriptionUtterance,
)


def _make_basic_result() -> TranscriptionResult:
    return TranscriptionResult(
        transcript=(
            "This is a sample transcript. We should follow up on action items. "
            "The team will deliver next week. Planning is a priority."
        ),
        duration=120.0,
        generated_at=datetime.now(),
        audio_file="/tmp/test.mp3",
        provider_name="Deepgram Nova 3",
        provider_features=["timestamps", "speaker_diarization", "topic_detection"],
    )


def test_full_analyzer_creates_all_files(tmp_path):
    result = _make_basic_result()
    # Add rich data
    result.summary = "Executive summary of the session."
    result.chapters = [
        TranscriptionChapter(
            start_time=0.0,
            end_time=60.0,
            topics=["intro", "planning"],
            confidence_scores=[0.9, 0.7],
        ),
        TranscriptionChapter(
            start_time=60.0, end_time=120.0, topics=["execution"], confidence_scores=[0.8]
        ),
    ]
    result.speakers = [
        TranscriptionSpeaker(id=0, total_time=70.0, percentage=58.3),
        TranscriptionSpeaker(id=1, total_time=50.0, percentage=41.7),
    ]
    result.utterances = [
        TranscriptionUtterance(speaker=0, start=5.0, end=15.0, text="This is a sample transcript."),
        TranscriptionUtterance(
            speaker=1, start=65.0, end=80.0, text="We should follow up on action items."
        ),
    ]
    result.topics = {"planning": 3, "execution": 2}
    result.intents = ["inform", "decide", "inform"]
    result.sentiment_distribution = {"positive": 3, "neutral": 1}

    analyzer = FullAnalyzer()
    analyzer.analyze_and_save(result, tmp_path, "video")

    # Assert all 5 files exist
    expected = [
        "01_executive_summary.md",
        "02_chapter_overview.md",
        "03_key_topics_and_intents.md",
        "04_full_transcript_with_timestamps.md",
        "05_key_insights_and_takeaways.md",
    ]
    for name in expected:
        f = tmp_path / name
        assert f.exists(), f"Missing expected file: {name}"
        content = f.read_text(encoding="utf-8")
        assert len(content) > 0


def test_full_analyzer_handles_missing_optionals(tmp_path):
    result = _make_basic_result()
    # No chapters, speakers, topics, intents, sentiments, utterances
    analyzer = FullAnalyzer()
    analyzer.analyze_and_save(result, tmp_path, "video")

    # Files exist and contain fallback copy
    summary = (tmp_path / "01_executive_summary.md").read_text(encoding="utf-8")
    assert "Executive Summary" in summary

    chapters = (tmp_path / "02_chapter_overview.md").read_text(encoding="utf-8")
    assert "No chapters identified" in chapters or "Chapter-by-Chapter Overview" in chapters

    topics = (tmp_path / "03_key_topics_and_intents.md").read_text(encoding="utf-8")
    assert "No specific topics" in topics or "Topic Frequency Analysis" in topics

    transcript_md = (tmp_path / "04_full_transcript_with_timestamps.md").read_text(encoding="utf-8")
    assert "Full Transcript" in transcript_md

    insights = (tmp_path / "05_key_insights_and_takeaways.md").read_text(encoding="utf-8")
    assert "Key Insights" in insights
