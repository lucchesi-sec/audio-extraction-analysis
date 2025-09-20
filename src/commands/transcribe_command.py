"""Transcription command implementation."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config.factory import ConfigFactory, ConfigType
from ..services.transcription import TranscriptionService
from ..ui.console import ConsoleManager
from ..utils.file_validation import ValidationError, validate_audio_file
from ..utils.paths import ensure_subpath, safe_write_json, sanitize_dirname

logger = logging.getLogger(__name__)


def transcribe_command(args: argparse.Namespace, console_manager: Optional[ConsoleManager] = None) -> int:
    """Handle the transcribe subcommand.

    Args:
        args: Command line arguments
        console_manager: Optional console manager for rich output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate input audio file
        try:
            input_path = validate_audio_file(args.audio_file)
        except ValidationError:
            return 1

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Safely build default transcript path: <stem>_transcript.txt
            output_path = input_path.parent / f"{input_path.stem}_transcript.txt"

        if console_manager:
            console_manager.setup_logging(logger)
            console_manager.print_stage("Transcription", "starting")
        logger.info(f"Transcribing {input_path} in {args.language} using {args.provider}")

        # Create transcription service (prefer class exported via src.cli for legacy patches)
        from .. import cli as cli_module

        service_cls = getattr(cli_module, "TranscriptionService", TranscriptionService)
        transcription_service = service_cls()

        # Transcribe audio
        result = transcription_service.transcribe(
            input_path,
            provider_name=args.provider if args.provider != "auto" else None,
            language=args.language,
        )

        if result:
            # Save result to file using service
            transcription_service.save_transcription_result(
                result, output_path, provider_name=result.provider_name
            )

            if console_manager:
                console_manager.print_stage("Transcription", "complete")
            logger.info("Transcription completed successfully")
            logger.info(f"Provider: {result.provider_name}")
            logger.info(f"Transcript length: {len(result.transcript):,} characters")
            logger.info(f"Duration: {result.duration:.1f} seconds")
            logger.info(f"Speakers detected: {len(result.speakers or [])}")
            logger.info(f"Output saved to: {output_path}")

            # Optional Markdown export if requested
            if getattr(args, "export_markdown", False):
                export_markdown_transcript(args, input_path, result)

            return 0
        else:
            logger.error("Transcription failed")
            return 1

    except Exception as e:
        logger.error(f"Transcribe command failed: {e}")
        return 1


def export_markdown_transcript(args: argparse.Namespace, input_path: Path, result: Any) -> None:
    """Export transcription result as Markdown.

    Args:
        args: Command line arguments
        input_path: Input audio file path
        result: Transcription result
    """
    try:
        from ..formatters.markdown_formatter import MarkdownFormatter

        out_root = Path(getattr(args, "markdown_output_dir", "output"))
        safe_name = sanitize_dirname(input_path.stem)
        base_dir = ensure_subpath(out_root, Path(safe_name))
        base_dir.mkdir(parents=True, exist_ok=True)

        md = MarkdownFormatter()
        source_info = {
            "source": str(input_path),
            "processed_at": datetime.now().isoformat(),
            "provider": result.provider_name,
            "total_duration": result.duration,
        }

        md_path = base_dir / "transcript.md"
        md_content = md.format_transcript(
            result,
            source_info,
            md_path,
            include_timestamps=getattr(args, "md_include_timestamps", True),
            include_speakers=getattr(args, "md_include_speakers", True),
            include_confidence=getattr(args, "md_include_confidence", False),
            template=getattr(args, "md_template", "default"),
        )
        md.save_transcript(md_content, md_path)

        # Save metadata.json
        metadata = {
            "source": source_info["source"],
            "processed_at": source_info["processed_at"],
            "provider": source_info["provider"],
            "duration_seconds": source_info["total_duration"],
            "segment_count": len(result.utterances or []),
        }
        try:
            safe_write_json(base_dir / "metadata.json", metadata)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed writing metadata.json: {e}")

        # Save segments.json
        segments = [
            {
                "text": getattr(u, "text", None) or getattr(u, "transcript", ""),
                "start_time": u.start,
                "end_time": u.end,
                "speaker": u.speaker,
            }
            for u in (result.utterances or [])
        ]
        try:
            safe_write_json(base_dir / "segments.json", segments)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed writing segments.json: {e}")

        logger.info(f"Markdown transcript saved to: {md_path}")

    except Exception as e:
        logger.error(f"Markdown export failed: {e}")
