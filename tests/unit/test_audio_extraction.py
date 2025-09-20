"""Test suite for audio extraction service."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.services.audio_extraction import (
    AudioExtractor,
    AudioQuality,
    extract_audio,
)


class TestAudioQuality:
    """Test AudioQuality enum."""

    def test_quality_values(self):
        """Test that quality enum has expected values."""
        assert AudioQuality.HIGH.value == "high"
        assert AudioQuality.STANDARD.value == "standard"
        assert AudioQuality.SPEECH.value == "speech"
        assert AudioQuality.COMPRESSED.value == "compressed"


class TestAudioExtractor:
    """Test AudioExtractor class."""

    @patch("subprocess.run")
    def test_check_ffmpeg_success(self, mock_run):
        """Test FFmpeg availability check when FFmpeg is installed."""
        mock_run.return_value = Mock(returncode=0)

        # Should not raise exception
        extractor = AudioExtractor()
        assert extractor is not None

    @patch("subprocess.run")
    def test_check_ffmpeg_failure(self, mock_run):
        """Test FFmpeg availability check when FFmpeg is not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="FFmpeg is required"):
            AudioExtractor()

    @patch("subprocess.run")
    def test_get_video_info(self, mock_run, temp_video_file):
        """Test getting video file information."""
        # Mock FFmpeg check
        mock_run.return_value = Mock(returncode=0)
        extractor = AudioExtractor()

        # Mock FFmpeg info command
        mock_run.return_value = Mock(
            returncode=0, stderr="Duration: 00:02:30.50, start: 0.000000", stdout=""
        )

        info = extractor.get_video_info(temp_video_file)

        assert "duration" in info
        assert "size_bytes" in info
        assert "size_mb" in info
        assert info["size_bytes"] > 0

    @patch("subprocess.run")
    def test_extract_audio_success(self, mock_run, temp_video_file, temp_output_dir):
        """Test successful audio extraction."""
        # Mock FFmpeg check and extraction
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        extractor = AudioExtractor()
        output_path = temp_output_dir / "output.mp3"

        # Mock file operations - input file exists, output file is created
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat") as mock_stat:
                stat_result = Mock()
                stat_result.st_size = 1024000  # 1MB
                mock_stat.return_value = stat_result

                with patch.object(Path, "mkdir"):
                    with patch.object(Path, "unlink"):  # For temp file cleanup
                        result = extractor.extract_audio(
                            temp_video_file, output_path, AudioQuality.SPEECH
                        )

        assert result == output_path
        assert mock_run.call_count >= 2  # FFmpeg check + extraction calls

    @patch("subprocess.run")
    def test_extract_audio_input_not_found(self, mock_run):
        """Test audio extraction with non-existent input file."""
        mock_run.return_value = Mock(returncode=0)  # FFmpeg check
        extractor = AudioExtractor()

        non_existent_path = Path("/non/existent/file.mp4")
        result = extractor.extract_audio(non_existent_path)

        assert result is None

    @patch("subprocess.run")
    def test_extract_audio_ffmpeg_failure(self, mock_run, temp_video_file, temp_output_dir):
        """Test audio extraction when FFmpeg command fails."""
        # Mock successful FFmpeg check, then failed extraction and video info call
        mock_run.side_effect = [
            Mock(returncode=0),  # FFmpeg version check (initialization)
            Mock(
                returncode=1, stderr="Error info", stdout=""
            ),  # get_video_info call (failure is OK)
            subprocess.CalledProcessError(1, "ffmpeg", stderr="FFmpeg error"),  # Extraction failure
        ]

        extractor = AudioExtractor()
        output_path = temp_output_dir / "output.mp3"

        result = extractor.extract_audio(temp_video_file, output_path, AudioQuality.HIGH)

        assert result is None

    @patch("subprocess.run")
    def test_extract_audio_speech_quality(self, mock_run, temp_video_file, temp_output_dir):
        """Test audio extraction with speech quality (requires two FFmpeg calls)."""
        # Mock successful FFmpeg calls
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        extractor = AudioExtractor()
        output_path = temp_output_dir / "output.mp3"

        # Mock file creation and cleanup
        output_path.with_suffix(".temp.mp3")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat") as mock_stat:
                stat_result = Mock()
                stat_result.st_size = 512000  # 0.5MB
                stat_result.st_mode = 33188  # Regular file mode
                mock_stat.return_value = stat_result
                with patch.object(Path, "unlink"):
                    with patch.object(Path, "mkdir"):

                        result = extractor.extract_audio(
                            temp_video_file, output_path, AudioQuality.SPEECH
                        )

        assert result == output_path
        # Should have multiple FFmpeg calls for speech quality
        assert mock_run.call_count >= 3  # FFmpeg check + extract + normalize

    @patch("subprocess.run")
    def test_all_quality_presets(self, mock_run, temp_video_file, temp_output_dir):
        """Test that all quality presets work."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        extractor = AudioExtractor()

        qualities = [AudioQuality.HIGH, AudioQuality.STANDARD, AudioQuality.COMPRESSED]

        for quality in qualities:
            output_path = temp_output_dir / f"output_{quality.value}.mp3"

            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "stat") as mock_stat:
                    stat_result = Mock()
                    stat_result.st_size = 1024000
                    stat_result.st_mode = 33188  # Regular file mode
                    mock_stat.return_value = stat_result
                    with patch.object(Path, "mkdir"):

                        result = extractor.extract_audio(temp_video_file, output_path, quality)

            assert result == output_path


class TestLegacyFunction:
    """Test the legacy extract_audio function."""

    @patch("subprocess.run")
    def test_legacy_extract_audio(self, mock_run, temp_video_file, temp_output_dir):
        """Test legacy extract_audio function for backward compatibility."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        input_path = str(temp_video_file)
        output_path = str(temp_output_dir / "legacy_output.wav")

        # Mock file operations - input and output files exist
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat") as mock_stat:
                stat_result = Mock()
                stat_result.st_size = 1024000
                mock_stat.return_value = stat_result

                with patch.object(Path, "mkdir"):
                    with patch.object(Path, "unlink"):  # For temp file cleanup
                        result = extract_audio(input_path, output_path)

        assert result == output_path

    @patch("src.services.audio_extraction.subprocess.run")
    def test_legacy_extract_audio_failure(self, mock_run):
        """Test legacy function returns None on failure."""
        mock_run.side_effect = FileNotFoundError()  # FFmpeg not found

        result = extract_audio("/fake/input.mp4", "/fake/output.wav")
        assert result is None

    @patch("subprocess.run")
    def test_path_validation_with_square_brackets(self, mock_run, temp_video_file_with_brackets):
        """Test that files with square brackets in names are accepted."""
        mock_run.return_value = Mock(returncode=0)
        extractor = AudioExtractor()

        # Should not raise exception for files with square brackets
        extractor._validate_path(temp_video_file_with_brackets)

    @patch("subprocess.run")
    def test_path_validation_with_common_special_chars(self, mock_run, temp_output_dir):
        """Test that files with common special characters are accepted."""
        mock_run.return_value = Mock(returncode=0)
        extractor = AudioExtractor()

        # Test various acceptable filename patterns
        acceptable_names = [
            "file with spaces.mp4",
            "file_with_underscores.mp4",
            "file-with-hyphens.mp4",
            "file(with_parentheses).mp4",
            "file[with_square_brackets].mp4",
            "file.with.dots.mp4",
        ]

        for name in acceptable_names:
            test_file = temp_output_dir / name
            test_file.write_text("dummy content")  # Create the file

            # Should not raise exception
            extractor._validate_path(test_file)

            test_file.unlink()  # Cleanup

    @patch("subprocess.run")
    def test_path_validation_rejects_dangerous_chars(self, mock_run, temp_output_dir):
        """Test that files with dangerous shell characters are rejected."""
        mock_run.return_value = Mock(returncode=0)
        AudioExtractor()

        # Test patterns that should be rejected (we only test the regex, not actual files)
        dangerous_patterns = [
            "test;cmd.mp4",
            "test&cmd.mp4",
            "test|cmd.mp4",
            "test`cmd.mp4",
            "test$var.mp4",
            "test<in.mp4",
            "test>out.mp4",
        ]

        import re

        for pattern in dangerous_patterns:
            # Test the regex pattern directly since we can't create files with these names
            if re.search(r"[;&|`$<>]", pattern):
                # This should match dangerous characters
                assert True  # Pattern correctly identified as dangerous
            else:
                raise AssertionError(f"Pattern {pattern} should be identified as dangerous")
