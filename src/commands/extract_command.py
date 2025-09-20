"""Audio extraction command implementation."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from ..services.audio_extraction import AudioExtractor, AudioQuality
from ..utils.sanitization import PathSanitizer
from ..ui.console import ConsoleManager

logger = logging.getLogger(__name__)


def extract_command(args: argparse.Namespace, console_manager: Optional[ConsoleManager] = None) -> int:
    """Handle the extract subcommand.

    Args:
        args: Command line arguments
        console_manager: Optional console manager for rich output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        input_path = Path(args.input_file)
        try:
            PathSanitizer.validate_path_security(input_path)
        except ValueError as exc:
            logger.error("Input file not found or invalid path")
            logger.debug("Path validation failure for %s: %s", input_path, exc)
            return 1

        allowed_suffixes = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".mkv", ".mov"}
        if input_path.suffix.lower() not in allowed_suffixes:
            logger.error("Invalid or unsupported input file type")
            logger.debug("Rejected file with suffix '%s'", input_path.suffix)
            return 1

        if not input_path.exists():
            logger.error("Input file not found or invalid path")
            logger.debug("Missing input path attempted: %s", input_path)
            return 1

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(".mp3")

        # Parse quality preset
        quality_map = {
            "high": AudioQuality.HIGH,
            "standard": AudioQuality.STANDARD,
            "speech": AudioQuality.SPEECH,
            "compressed": AudioQuality.COMPRESSED,
        }

        quality = quality_map.get(args.quality, AudioQuality.SPEECH)

        display_name = input_path.name

        if console_manager:
            console_manager.setup_logging(logger)
            console_manager.print_stage("Audio Extraction", "starting")
        logger.info(
            "Extracting audio from %s with %s quality",
            display_name,
            getattr(quality, "value", quality),
        )
        logger.debug("Full input path: %s", input_path)

        # Extract audio (prefer class referenced from src.cli for backward compatibility)
        from .. import cli as cli_module

        extractor_cls = getattr(cli_module, "AudioExtractor", AudioExtractor)

        extractor = extractor_cls()
        if console_manager:
            with console_manager.progress_context("Extracting audio...") as progress:
                progress.update(10)
                result_path = extractor.extract_audio(input_path, output_path, quality)
                progress.update(100)
        else:
            result_path = extractor.extract_audio(input_path, output_path, quality)

        if result_path:
            if console_manager:
                console_manager.print_stage("Audio Extraction", "complete")
            logger.info(f"Audio extracted successfully: {result_path}")
            return 0
        else:
            logger.error("Audio extraction failed")
            return 1

    except Exception as e:
        logger.error(f"Extract command failed: {e}")
        return 1
