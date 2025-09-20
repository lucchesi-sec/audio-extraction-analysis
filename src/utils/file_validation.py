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
        # Convert to Path if string
        file_path = Path(audio_file_path)
        
        # Apply provider-specific size limits if known
        if provider_name and not max_file_size:
            provider_limits = {
                'elevenlabs': 50 * 1024 * 1024,  # 50MB
                'deepgram': 2 * 1024 * 1024 * 1024,  # 2GB  
            }
            max_file_size = provider_limits.get(provider_name.lower())
        
        # Use existing FileValidator for comprehensive validation
        FileValidator.validate_audio_file(
            file_path,
            max_file_size=max_file_size,
            must_exist=True
        )
        
        return file_path
        
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {audio_file_path}")
        raise ValidationError(f"Audio file not found: {audio_file_path}") from e
    except PermissionError as e:
        logger.error(f"Permission denied accessing file: {audio_file_path}")
        raise ValidationError(f"Cannot access file: {audio_file_path}") from e
    except ValueError as e:
        logger.error(f"Invalid audio file: {e}")
        raise ValidationError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise ValidationError(f"Validation failed: {e}") from e


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
        # Convert to Path if string
        file_path = Path(media_file_path)
        
        # Use existing FileValidator for comprehensive validation
        FileValidator.validate_media_file(
            file_path,
            max_size=max_file_size
        )
        
        return file_path
        
    except FileNotFoundError as e:
        logger.error(f"Media file not found: {media_file_path}")
        raise ValidationError(f"Media file not found: {media_file_path}") from e
    except PermissionError as e:
        logger.error(f"Permission denied accessing file: {media_file_path}")
        raise ValidationError(f"Cannot access file: {media_file_path}") from e
    except ValueError as e:
        logger.error(f"Invalid media file: {e}")
        raise ValidationError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise ValidationError(f"Validation failed: {e}") from e


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
