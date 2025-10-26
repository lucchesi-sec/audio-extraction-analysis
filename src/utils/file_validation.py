"""File validation utility to consolidate duplicate validation patterns.

This module provides centralized validation functions that replace
the scattered file existence checks throughout the codebase.

The module offers two validation styles:
1. **Standard validators** (validate_*): Raise ValidationError on failure
2. **Safe validators** (safe_validate_*): Return None on failure

Use standard validators when you want explicit error handling and detailed
exception information. Use safe validators when you prefer None-checking
over exception handling, particularly for optional file processing.

Provider-specific size limits:
- ElevenLabs: 50MB
- Deepgram: 2GB
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .validation import FileValidator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures.

    This exception wraps underlying errors (FileNotFoundError, PermissionError,
    ValueError) to provide a consistent error handling interface. The original
    exception is preserved in the exception chain for debugging.

    All validation functions in this module raise ValidationError on failure,
    allowing calling code to catch a single exception type while still having
    access to the underlying cause via exception chaining.
    """
    pass


def _get_provider_size_limit(provider_name: str) -> Optional[int]:
    """Get provider-specific file size limit.

    This function returns known size limits for audio processing providers.
    The lookup is case-insensitive for convenience.

    Args:
        provider_name: Name of the provider service (e.g., 'elevenlabs', 'deepgram').
                      Case-insensitive.

    Returns:
        File size limit in bytes, or None if provider not recognized.
        Known limits: ElevenLabs (50MB), Deepgram (2GB).

    Example:
        >>> _get_provider_size_limit('elevenlabs')
        52428800  # 50MB in bytes
        >>> _get_provider_size_limit('ElevenLabs')  # Case insensitive
        52428800
        >>> _get_provider_size_limit('unknown')
        None
    """
    # Provider-specific size limits based on official API documentation
    # These limits are enforced by the respective services
    provider_limits = {
        'elevenlabs': 50 * 1024 * 1024,  # 50MB - ElevenLabs API limit
        'deepgram': 2 * 1024 * 1024 * 1024,  # 2GB - Deepgram API limit
    }
    return provider_limits.get(provider_name.lower())


def _handle_validation_exception(
    e: Exception,
    file_path: Path | str,
    file_type: str = 'audio'
) -> None:
    """Handle validation exceptions with appropriate logging and error wrapping.

    This function centralizes exception handling for validation operations by:
    1. Logging the error with appropriate severity
    2. Wrapping the original exception in ValidationError for consistent handling
    3. Preserving the exception chain for debugging (using 'from e')

    The function logs all errors and then re-raises them as ValidationError,
    ensuring that all validation failures have a consistent exception type
    while preserving the original error information.

    Args:
        e: The exception to handle (FileNotFoundError, PermissionError,
           ValueError, or other Exception)
        file_path: Path to the file being validated (for error messages)
        file_type: Type of file for error messages ('audio' or 'media')

    Raises:
        ValidationError: Always raised, wrapping the original exception.
                        The original exception is accessible via exception chaining.
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

    The function performs comprehensive validation including existence,
    accessibility, format checking, and optional size validation. If a
    provider_name is specified, it automatically applies provider-specific
    size limits (e.g., 50MB for ElevenLabs, 2GB for Deepgram).

    Args:
        audio_file_path: Path to audio file (Path or string). Converted to
                        Path object internally.
        max_file_size: Optional maximum file size in bytes. If not specified
                      and provider_name is given, uses provider-specific limit.
        provider_name: Optional provider name for automatic size limits
                      (e.g., 'elevenlabs', 'deepgram'). Case-insensitive.

    Returns:
        Path object if validation passes.

    Raises:
        ValidationError: Wraps all validation failures. The underlying cause
                        may be FileNotFoundError (file doesn't exist),
                        PermissionError (file not accessible), or ValueError
                        (invalid format or too large). Access the original
                        exception via exception chaining (__cause__).

    Example:
        >>> validate_audio_file('audio.mp3')
        PosixPath('audio.mp3')
        >>> validate_audio_file('audio.mp3', provider_name='elevenlabs')
        PosixPath('audio.mp3')  # Validates with 50MB limit
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

    This function validates both audio and video files, making it particularly
    useful for audio extraction workflows where the input may be either format.
    It performs comprehensive validation including existence, accessibility,
    format checking, and optional size validation.

    Args:
        media_file_path: Path to media file (Path or string). Supports both
                        audio formats (mp3, wav, flac, etc.) and video formats
                        (mp4, avi, mkv, etc.). Converted to Path object internally.
        max_file_size: Optional maximum file size in bytes. No limit if not specified.

    Returns:
        Path object if validation passes.

    Raises:
        ValidationError: Wraps all validation failures. The underlying cause
                        may be FileNotFoundError (file doesn't exist),
                        PermissionError (file not accessible), or ValueError
                        (invalid format or too large). Access the original
                        exception via exception chaining (__cause__).

    Example:
        >>> validate_media_file('video.mp4')
        PosixPath('video.mp4')
        >>> validate_media_file('audio.mp3', max_file_size=10*1024*1024)
        PosixPath('audio.mp3')  # Validates with 10MB limit
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

    This function provides a None-returning interface for media file validation,
    useful when you prefer None-checking over exception handling. It's particularly
    convenient for optional file processing or when validation failure should be
    handled as a normal case rather than an exceptional condition.

    Use this function when:
    - Processing optional media files
    - Filtering lists of potential media files
    - Implementing fallback logic for missing files
    - Simplifying error handling in data pipelines

    Args:
        media_file_path: Path to media file (Path or string). Supports both
                        audio and video formats.
        max_file_size: Optional maximum file size in bytes.

    Returns:
        Path object if validation passes, None if validation fails for any reason.

    Example:
        >>> result = safe_validate_media_file('video.mp4')
        >>> if result:
        ...     process_media(result)
        ... else:
        ...     logger.warning('Media file validation failed')
        >>>
        >>> # Filter valid files from a list
        >>> files = ['a.mp4', 'b.mp3', 'missing.wav']
        >>> valid = [f for f in files if safe_validate_media_file(f)]
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
    """Safe wrapper for audio file validation that returns None instead of raising exceptions.

    This function provides a None-returning interface for audio file validation,
    useful when you prefer None-checking over exception handling. It's particularly
    convenient for optional file processing or when validation failure should be
    handled as a normal case rather than an exceptional condition.

    Use this function when:
    - Processing optional audio files
    - Filtering lists of potential audio files
    - Implementing fallback logic for missing files
    - Simplifying error handling in data pipelines
    - Working with provider-specific validation where failures are common

    Args:
        audio_file_path: Path to audio file (Path or string).
        max_file_size: Optional maximum file size in bytes. If not specified
                      and provider_name is given, uses provider-specific limit.
        provider_name: Optional provider name for automatic size limits
                      (e.g., 'elevenlabs', 'deepgram'). Case-insensitive.

    Returns:
        Path object if validation passes, None if validation fails for any reason.

    Example:
        >>> result = safe_validate_audio_file('audio.mp3', provider_name='elevenlabs')
        >>> if result:
        ...     transcribe_with_elevenlabs(result)
        ... else:
        ...     logger.warning('Audio file failed ElevenLabs validation (possibly >50MB)')
        >>>
        >>> # Try multiple providers with different size limits
        >>> audio_path = 'large_audio.mp3'
        >>> if safe_validate_audio_file(audio_path, provider_name='deepgram'):
        ...     use_deepgram(audio_path)  # Accepts up to 2GB
        >>> elif safe_validate_audio_file(audio_path, provider_name='elevenlabs'):
        ...     use_elevenlabs(audio_path)  # Accepts up to 50MB
    """
    try:
        return validate_audio_file(audio_file_path, max_file_size, provider_name)
    except ValidationError:
        return None
