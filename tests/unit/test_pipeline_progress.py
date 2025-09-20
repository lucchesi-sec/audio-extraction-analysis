"""Integration tests for pipeline progress tracking."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.audio_pipeline import AudioProcessingPipeline
from src.ui.console import ConsoleManager


@pytest.mark.asyncio
async def test_pipeline_with_progress():
    """Test complete pipeline with progress tracking using JSON mode for stability."""
    console_manager = ConsoleManager(json_output=True)
    pipeline = AudioProcessingPipeline(console_manager=console_manager)

    with patch("src.pipeline.audio_pipeline.AudioExtractor") as mock_extractor, patch(
        "src.pipeline.audio_pipeline.TranscriptionService"
    ) as mock_transcriber, patch("src.pipeline.audio_pipeline.FullAnalyzer") as mock_analyzer:

        # Setup mocks
        mock_extractor_instance = MagicMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_audio.return_value = "test_audio.wav"

        mock_transcriber_instance = AsyncMock()
        mock_transcriber.return_value = mock_transcriber_instance
        # Return a simple object to carry through to analyzer
        mock_transcriber_instance.transcribe_async.return_value = MagicMock()

        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance
        mock_analyzer_instance.analyze_and_save.return_value = {"a": Path("analysis_file.md")}

        # Execute
        with tempfile.TemporaryDirectory() as temp_dir:
            results = await pipeline.process_file("test_input.mp4", temp_dir, analysis_style="full")

        assert "audio_path" in results
        assert "transcript" in results
        assert "analysis_files" in results
        assert "stage_results" in results

        # Verify stages
        assert results["stage_results"]["extraction"]["status"] == "complete"
        assert results["stage_results"]["transcription"]["status"] == "complete"
        assert results["stage_results"]["analysis"]["status"] == "complete"
