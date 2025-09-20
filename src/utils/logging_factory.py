"""Centralized logging factory for consistent logger creation across the application."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class LoggingFactory:
    """Factory for creating and configuring loggers consistently."""

    _initialized = False
    _log_dir = Path("logs")

    @classmethod
    def initialize(
        cls,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
    ) -> None:
        """Initialize the logging system once for the entire application.

        Args:
            log_dir: Directory for log files (default: "logs")
            level: Default logging level
            format_string: Custom format string for log messages
        """
        if cls._initialized:
            return

        if log_dir:
            cls._log_dir = log_dir

        # Create log directory if it doesn't exist
        cls._log_dir.mkdir(exist_ok=True)

        # Default format
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure root logger
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.FileHandler(cls._log_dir / "app.log"), logging.StreamHandler()],
        )

        # Set specific module levels
        logging.getLogger("transcription").setLevel(logging.DEBUG)
        logging.getLogger("audio_extraction").setLevel(logging.INFO)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a configured logger for the given module name.

        Args:
            name: Module name for the logger

        Returns:
            Configured logger instance
        """
        # Auto-initialize if not done yet
        if not cls._initialized:
            cls.initialize()

        return logging.getLogger(name)

    @classmethod
    def set_level(cls, name: str, level: int) -> None:
        """Set the logging level for a specific logger.

        Args:
            name: Logger name
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        logging.getLogger(name).setLevel(level)

    @classmethod
    def configure_verbose(cls, verbose: bool = False) -> None:
        """Configure verbosity for all loggers.

        Args:
            verbose: If True, set to DEBUG level; otherwise INFO
        """
        level = logging.DEBUG if verbose else logging.INFO
        logging.getLogger().setLevel(level)
        logging.getLogger("src").setLevel(level)

        # Update specific loggers based on verbosity
        if verbose:
            logging.getLogger("transcription").setLevel(logging.DEBUG)
            logging.getLogger("audio_extraction").setLevel(logging.DEBUG)
        else:
            logging.getLogger("transcription").setLevel(logging.INFO)
            logging.getLogger("audio_extraction").setLevel(logging.INFO)


# Convenience function for backward compatibility
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name.

    This function maintains backward compatibility with the old logging_config module.

    Args:
        name: Module name for the logger

    Returns:
        Configured logger instance
    """
    return LoggingFactory.get_logger(name)
