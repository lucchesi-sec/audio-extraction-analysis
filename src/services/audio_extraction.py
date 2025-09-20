"""Audio extraction service using FFmpeg."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.common_validation import FileValidator
from ..utils.file_validation import safe_validate_media_file
from .ffmpeg_core import build_extract_commands

logger = logging.getLogger(__name__)


class AudioQuality(Enum):
    """Audio quality presets for extraction."""

    HIGH = "high"  # 320k bitrate - Best for archival
    STANDARD = "standard"  # Variable bitrate - Good balance
    SPEECH = "speech"  # Mono, normalized - Best for transcription
    COMPRESSED = "compressed"  # 128k - Smaller file size


class AudioExtractor:
    """FFmpeg-based audio extraction service."""

    # Security: Define allowed file extensions and maximum file size
    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"}
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB limit

    def __init__(self):
        self._check_ffmpeg()

    def _validate_path(self, file_path: Path) -> None:
        """Validate file path for security.

        Args:
            file_path: Path to validate

        Raises:
            ValueError: If path is invalid or potentially dangerous
        """
        # Delegate to common validation utility
        FileValidator.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=self.ALLOWED_EXTENSIONS,
            max_size=self.MAX_FILE_SIZE
        )

        # Security: Check for path traversal attempts
        resolved_path = file_path.resolve()
        if ".." in str(resolved_path) or str(resolved_path).startswith("/"):
            # Allow absolute paths but validate they don't contain dangerous shell characters
            # Note: Square brackets [], parentheses (), and spaces are common in media filenames
            # and are safe when properly quoted with shlex.quote()
            path_str = str(resolved_path)
            if re.search(r"[;&|`$<>]", path_str):
                raise ValueError(f"Invalid characters in file path: {file_path}")

    def _sanitize_path(self, file_path: Path) -> str:
        """Sanitize file path for safe subprocess usage.

        Args:
            file_path: Path to sanitize

        Returns:
            Safely quoted path string
        """
        return shlex.quote(str(file_path.resolve()))

    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg check failed")
            raise RuntimeError(
                "FFmpeg is required but not installed. "
                "Install with: brew install ffmpeg (macOS) or "
                "sudo apt-get install ffmpeg (Ubuntu)"
            ) from e
        except FileNotFoundError as e:
            logger.error("FFmpeg is not installed or not accessible")
            raise RuntimeError(
                "FFmpeg is required but not installed. "
                "Install with: brew install ffmpeg (macOS) or "
                "sudo apt-get install ffmpeg (Ubuntu)"
            ) from e
        except subprocess.TimeoutExpired as e:
            logger.error("FFmpeg version check timed out")
            raise RuntimeError("FFmpeg version check timed out") from e

    def get_video_info(self, input_path: Path) -> Dict[str, Any]:
        """Get video file information."""
        try:
            # Security: Validate input path
            self._validate_path(input_path)

            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-f",
                "null",
                "-",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Extract duration from stderr (FFmpeg logs to stderr)
            duration_line = [line for line in result.stderr.split("\n") if "Duration:" in line]
            duration = None
            if duration_line:
                duration = duration_line[0].split("Duration: ")[1].split(",")[0]

            file_size = input_path.stat().st_size

            return {
                "duration": duration,
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024),
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"FFmpeg failed to get video info: {e}")
            return {}
        except (FileNotFoundError, ValueError, PermissionError) as e:
            logger.warning(f"Could not get video info: {e}")
            return {}
        except OSError as e:
            logger.warning(f"System error getting video info: {e}")
            return {}

    def extract_audio(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        quality: AudioQuality = AudioQuality.SPEECH,
    ) -> Optional[Path]:
        """Extract audio from video using specified quality preset.

        Args:
            input_path: Input video file path
            output_path: Output audio file path (optional)
            quality: Audio quality preset

        Returns:
            Path to extracted audio file, or None if extraction failed
        """
        # Validate input video file (for audio extraction)
        validated_path = safe_validate_media_file(
            input_path, 
            max_file_size=self.MAX_FILE_SIZE
        )
        if validated_path is None:
            return None
        input_path = validated_path

        # Set default output path
        if output_path is None:
            output_path = input_path.with_suffix(".mp3")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting audio from {input_path} with {quality.value} quality")

        try:
            # Get video info for logging
            info = self.get_video_info(input_path)
            if info:
                logger.info(
                    f"Input video: {info.get('duration', 'unknown')} duration, "
                    f"{info.get('size_mb', 0):.2f} MB"
                )

            cmds, temp_path = build_extract_commands(input_path, output_path, quality.value)

            # Run FFmpeg command(s)
            for cmd in cmds:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)

            # Log success
            if output_path.exists():
                final_size = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"Successfully extracted audio: {final_size:.2f} MB")
                return output_path
            else:
                logger.error("Audio extraction completed but output file not found")
                return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
        except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
        finally:
            # Cleanup possible temp file from SPEECH pipeline
            try:
                if 'temp_path' in locals() and temp_path and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass


# ---------------------- Legacy shim for backward compatibility ----------------------
def extract_audio(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    quality: AudioQuality = AudioQuality.SPEECH,
) -> Optional[str]:
    """Legacy function wrapper for audio extraction.

    This preserves the historical module-level API expected by older code and tests.

    Args:
        input_path: Input video file path
        output_path: Optional output audio path
        quality: Audio quality preset

    Returns:
        Path to extracted audio file, or None on failure
    """
    try:
        extractor = AudioExtractor()
        in_path = Path(input_path)
        out_path = Path(output_path) if output_path is not None else None
        res = extractor.extract_audio(in_path, out_path, quality)
        return str(res) if res is not None else None
    except Exception:
        # Match legacy behavior: swallow and return None
        return None
