"""File validation utility to consolidate duplicate validation patterns.

This module provides centralized validation functions that replace
the scattered file existence checks throughout the codebase.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .validation import FileValidator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def _get_provider_size_limit(provider_name: str) -> Optional[int]:
    """Get provider-specific file size limit.

    Args:
        provider_name: Name of the provider service

    Returns:
        File size limit in bytes, or None if provider not recognized
    """
    provider_limits = {
        'elevenlabs': 50 * 1024 * 1024,  # 50MB
        'deepgram': 2 * 1024 * 1024 * 1024,  # 2GB
    }
    return provider_limits.get(provider_name.lower())


def _handle_validation_exception(
    e: Exception,
    file_path: Path | str,
    file_type: str = 'audio'
) -> None:
    """Handle validation exceptions with appropriate logging and error wrapping.

    Args:
        e: The exception to handle
        file_path: Path to the file being validated
        file_type: Type of file for error messages ('audio' or 'media')

    Raises:
        ValidationError: Wrapped exception with context
    """
    if isinstance(e, FileNotFoundError):
        logger.error(f"{file_type.capitalize()} file not found: {file_path}")
        raise ValidationError(f"{file_type.capitalize()} file not found: {file_path}") from e
    elif isinstance(e, PermissionError):
        logger.error(f"Permission denied accessing file: {file_path}")
        raise ValidationError(f"Cannot access file: {file_path}") from e
    elif isinstance(e, ValueError):
        logger.error(f"Invalid {file_type} file: {e}")
        raise ValidationError(str(e)) from e
    else:
        logger.error(f"Unexpected validation error: {e}")
        raise ValidationError(f"Validation failed: {e}") from e


def validate_audio_file(
    audio_file_path: Path | str,
    max_file_size: Optional[int] = None,
    provider_name: Optional[str] = None
) -> Path:
    """Validate an audio file exists and is accessible.

    This function consolidates the duplicate validation pattern:
    ```python
    if not audio_file_path.exists():
        logger.error(f"Audio file not found: {audio_file_path}")
        return None
    ```

    Args:
        audio_file_path: Path to audio file (Path or string)
        max_file_size: Optional maximum file size in bytes
        provider_name: Optional provider name for specific size limits

    Returns:
        Path object if validation passes

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not accessible
        ValueError: If file is invalid format or too large
    """
    try:
        file_path = Path(audio_file_path)

        # Apply provider-specific size limits if known
        if provider_name and not max_file_size:
            max_file_size = _get_provider_size_limit(provider_name)

        # Use existing FileValidator for comprehensive validation
        FileValidator.validate_audio_file(
            file_path,
            max_file_size=max_file_size,
            must_exist=True
        )

        return file_path

    except Exception as e:
        _handle_validation_exception(e, audio_file_path, 'audio')


def validate_media_file(
    media_file_path: Path | str,
    max_file_size: Optional[int] = None
) -> Path:
    """Validate a media file (audio or video) exists and is accessible.

    This function validates both audio and video files, useful for audio extraction.

    Args:
        media_file_path: Path to media file (Path or string)
        max_file_size: Optional maximum file size in bytes

    Returns:
        Path object if validation passes

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not accessible
        ValueError: If file is invalid format or too large
    """
    try:
        file_path = Path(media_file_path)

        # Use existing FileValidator for comprehensive validation
        FileValidator.validate_media_file(
            file_path,
            max_size=max_file_size
        )

        return file_path

    except Exception as e:
        _handle_validation_exception(e, media_file_path, 'media')


def safe_validate_media_file(
    media_file_path: Path | str,
    max_file_size: Optional[int] = None
) -> Optional[Path]:
    """Safe wrapper for media file validation that returns None instead of raising exceptions.
    
    Args:
        media_file_path: Path to media file
        max_file_size: Optional maximum file size in bytes
        
    Returns:
        Path object if validation passes, None if validation fails
    """
    try:
        return validate_media_file(media_file_path, max_file_size)
    except ValidationError:
        return None


def safe_validate_audio_file(
    audio_file_path: Path | str,
    max_file_size: Optional[int] = None,
    provider_name: Optional[str] = None
) -> Optional[Path]:
    """Safe wrapper that returns None instead of raising exceptions.
    
    This is useful for cases where the calling code expects None on failure
    rather than catching exceptions.
    
    Args:
        audio_file_path: Path to audio file
        max_file_size: Optional maximum file size in bytes
        provider_name: Optional provider name for specific limits
        
    Returns:
        Path object if validation passes, None if validation fails
    """
    try:
        return validate_audio_file(audio_file_path, max_file_size, provider_name)
    except ValidationError:
        return None
