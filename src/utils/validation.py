"""Centralized file validation utilities for security and integrity checks.

This module now delegates to common_validation for consistency.
Maintained for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

# Import from common validation for consistency
from .common_validation import FileValidator as CommonFileValidator
from .sanitization import PathSanitizer


class FileValidator(CommonFileValidator):
    """Unified file validation for security and format checks.
    
    This class extends CommonFileValidator to maintain backward compatibility
    while using the centralized implementation.
    """

    # Re-export constants for backward compatibility
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}

    @classmethod
    def validate_path_security(cls, file_path: Path) -> None:
        """Validate a path for security issues.

        Args:
            file_path: Path to validate

        Raises:
            ValueError: If path contains dangerous characters
        """
        # Delegate to PathSanitizer for consistency
        PathSanitizer.validate_path_security(file_path)

    @classmethod
    def validate_video_file(cls, file_path: Path, max_file_size: Optional[int] = None) -> None:
        """Validate a video file path.

        Args:
            file_path: Path to video file
            max_file_size: Maximum file size in bytes (default: 2GB)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
        """
        cls.validate_file_path(
            file_path,
            allowed_extensions=cls.VIDEO_EXTENSIONS,
            max_size=max_file_size or cls.DEFAULT_MAX_FILE_SIZE,
            must_exist=True,
        )

    @classmethod
    def validate_audio_file(
        cls, file_path: Path, max_file_size: Optional[int] = None, must_exist: bool = True
    ) -> None:
        """Validate an audio file path.

        Args:
            file_path: Path to audio file
            max_file_size: Maximum file size in bytes (default: 2GB)
            must_exist: Whether the file must exist

        Raises:
            FileNotFoundError: If file doesn't exist and must_exist is True
            ValueError: If validation fails
        """
        cls.validate_file_path(
            file_path,
            allowed_extensions=cls.AUDIO_EXTENSIONS,
            max_size=max_file_size or cls.DEFAULT_MAX_FILE_SIZE,
            must_exist=must_exist,
        )

    @classmethod
    def is_valid_extension(cls, file_path: Path, extensions: Set[str]) -> bool:
        """Check if a file has a valid extension.

        Args:
            file_path: Path to check
            extensions: Set of valid extensions

        Returns:
            True if extension is valid, False otherwise
        """
        return file_path.suffix.lower() in extensions

    @classmethod
    def get_file_size_mb(cls, file_path: Path) -> float:
        """Get file size in megabytes.

        Args:
            file_path: Path to file

        Returns:
            File size in MB, or 0.0 if file doesn't exist
        """
        try:
            if file_path.exists():
                return file_path.stat().st_size / (1024 * 1024)
        except (OSError, PermissionError):
            pass
        return 0.0
