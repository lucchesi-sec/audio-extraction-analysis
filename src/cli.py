"""Unified CLI for audio extraction and transcription analysis.

This module serves as the main entry point for the CLI and delegates
command execution to the refactored command modules.
"""
from __future__ import annotations

import logging
import sys

# Import from refactored command modules
try:
    from .commands import (
        create_parser,
        export_markdown_command,
        extract_command,
        process_command,
        setup_logging,
        transcribe_command,
    )
    from .config.config import Config  # Backwards compatibility for tests
    from .pipeline.audio_pipeline import AudioProcessingPipeline  # Backward compatibility for tests
    from .services.audio_extraction import AudioExtractor  # Backwards compatibility for tests
    from .services.transcription import TranscriptionService  # Backwards compatibility for tests
    from .ui.console import ConsoleManager
except ImportError:  # pragma: no cover - fallback for installed package layout
    from commands import (
        create_parser,
        export_markdown_command,
        extract_command,
        process_command,
        setup_logging,
        transcribe_command,
    )
    from config.config import Config
    from pipeline.audio_pipeline import AudioProcessingPipeline
    from services.audio_extraction import AudioExtractor
    from services.transcription import TranscriptionService
    from ui.console import ConsoleManager

logger = logging.getLogger(__name__)


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Setup console manager if not in JSON output mode
    console_manager = None
    if not args.json_output:
        console_manager = ConsoleManager(verbose=args.verbose)

    try:
        # Route to appropriate command handler
        if args.command == "extract":
            return extract_command(args, console_manager)
        elif args.command == "transcribe":
            return transcribe_command(args, console_manager)
        elif args.command == "process":
            return process_command(args, console_manager)
        elif args.command == "export-markdown":
            # Handle export-markdown command
            return export_markdown_command(args, console_manager)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logger.error("Operation cancelled by user")
        return 1



if __name__ == "__main__":
    sys.exit(main())
