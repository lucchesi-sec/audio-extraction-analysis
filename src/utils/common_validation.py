"""Common validation utilities for the audio extraction pipeline.

This module consolidates validation logic that was previously duplicated
across multiple modules, providing a single source of truth for validation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

from .sanitization import PathSanitizer

logger = logging.getLogger(__name__)


class FileValidator:
    """Centralized file validation utilities."""
    
    # Common audio/video extensions
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    
    # Default size limits
    DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    
    @classmethod
    def _check_file_existence(cls, file_path: Path) -> None:
        """Check if file exists.

        Args:
            file_path: Path to check

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @classmethod
    def _check_file_extension(cls, file_path: Path, allowed_extensions: Set[str]) -> None:
        """Check if file extension is allowed.

        Args:
            file_path: Path to check
            allowed_extensions: Set of allowed file extensions (with dots)

        Raises:
            ValueError: If extension is not allowed
        """
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Allowed: {', '.join(sorted(allowed_extensions))}"
            )

    @classmethod
    def _check_file_size(cls, file_path: Path, max_size: int) -> None:
        """Check if file size is within limits.

        Args:
            file_path: Path to check
            max_size: Maximum file size in bytes

        Raises:
            ValueError: If file exceeds size limit
        """
        file_size = file_path.stat().st_size
        if file_size > max_size:
            raise ValueError(
                f"File size {file_size:,} bytes exceeds maximum {max_size:,} bytes"
            )

    @classmethod
    def _check_file_type(cls, file_path: Path) -> None:
        """Check if path is a regular file.

        Args:
            file_path: Path to check

        Raises:
            ValueError: If path is not a file
        """
        try:
            isf = file_path.is_file()
        except Exception:  # e.g., mocked stat without st_mode
            isf = True  # Defer to permission check
        if not isf:
            raise ValueError(f"Path is not a file: {file_path}")

    @classmethod
    def _check_file_permissions(cls, file_path: Path) -> None:
        """Check if file is readable.

        Args:
            file_path: Path to check

        Raises:
            PermissionError: If file cannot be read
        """
        try:
            with open(file_path, 'rb'):
                pass
        except PermissionError as e:
            raise PermissionError(f"Cannot read file: {file_path}") from e

    @classmethod
    def validate_file_path(
        cls,
        file_path: Path,
        must_exist: bool = True,
        allowed_extensions: Optional[Set[str]] = None,
        max_size: Optional[int] = None
    ) -> None:
        """Validate a file path with comprehensive checks.

        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            allowed_extensions: Set of allowed file extensions (with dots)
            max_size: Maximum file size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist and must_exist is True
            ValueError: If validation fails
            PermissionError: If file is not readable
        """
        file_path = Path(file_path)

        # Security validation - delegated to sanitizer
        PathSanitizer.validate_path_security(file_path)

        # Run existence-dependent checks
        if must_exist:
            cls._check_file_existence(file_path)
            cls._check_file_type(file_path)
            cls._check_file_permissions(file_path)

        # Extension check (can run regardless of existence)
        if allowed_extensions:
            cls._check_file_extension(file_path, allowed_extensions)

        # Size check (requires file to exist)
        if must_exist and max_size is not None:
            cls._check_file_size(file_path, max_size)
    
    @classmethod
    def validate_media_file(
        cls,
        file_path: Path,
        max_size: Optional[int] = None
    ) -> None:
        """Validate a media file (audio or video).
        
        Args:
            file_path: Path to media file
            max_size: Maximum file size in bytes (default: 2GB)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If not a valid media file
        """
        cls.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=cls.MEDIA_EXTENSIONS,
            max_size=max_size or cls.DEFAULT_MAX_FILE_SIZE
        )
    
    @classmethod
    def validate_output_path(
        cls,
        output_path: Path,
        force: bool = False,
        create_parents: bool = True
    ) -> None:
        """Validate an output file path.
        
        Args:
            output_path: Path for output file
            force: Whether to allow overwriting existing files
            create_parents: Whether to create parent directories
            
        Raises:
            ValueError: If output path is invalid
            FileExistsError: If file exists and force is False
        """
        output_path = Path(output_path)
        
        # Security validation
        PathSanitizer.validate_path_security(output_path)
        
        # Check if file exists
        if output_path.exists() and not force:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Use force=True to overwrite."
            )
        
        # Create parent directories if needed
        if create_parents:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        elif not output_path.parent.exists():
            raise ValueError(f"Output directory does not exist: {output_path.parent}")
        
        # Check write permissions on parent directory
        if not output_path.parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {output_path.parent}")
            
        # Test write permissions
        test_file = output_path.parent / f".write_test_{output_path.name}"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Cannot write to directory: {output_path.parent}"
            ) from e


class ConfigValidator:
    """Validation for configuration values."""
    
    @staticmethod
    def validate_positive_number(value: float, name: str) -> None:
        """Validate that a value is a positive number.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def validate_range(
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "Value"
    ) -> None:
        """Validate that a value is within a range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is outside the range
        """
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be at least {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be at most {max_val}, got {value}")
    
    @staticmethod
    def validate_enum(value: str, allowed: Set[str], name: str = "Value") -> None:
        """Validate that a value is in an allowed set.
        
        Args:
            value: Value to validate
            allowed: Set of allowed values
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is not in allowed set
        """
        if value not in allowed:
            raise ValueError(
                f"{name} must be one of {sorted(allowed)}, got '{value}'"
            )


# Convenience functions for backward compatibility
def validate_file_path(file_path: Path, **kwargs) -> None:
    """Validate a file path. See FileValidator.validate_file_path for details."""
    FileValidator.validate_file_path(file_path, **kwargs)


def validate_media_file(file_path: Path, **kwargs) -> None:
    """Validate a media file. See FileValidator.validate_media_file for details."""
    FileValidator.validate_media_file(file_path, **kwargs)


def validate_output_path(output_path: Path, **kwargs) -> None:
    """Validate an output path. See FileValidator.validate_output_path for details."""
    FileValidator.validate_output_path(output_path, **kwargs)
