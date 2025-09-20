"""Export Markdown transcript command implementation."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..formatters.markdown_formatter import MarkdownFormatter
from ..services.transcription import TranscriptionService
from ..utils.paths import ensure_subpath, safe_write_json, sanitize_dirname
from ..utils.file_validation import ValidationError, validate_audio_file
from ..ui.console import ConsoleManager

logger = logging.getLogger(__name__)


def export_markdown_command(args, console_manager: Optional[ConsoleManager] = None) -> int:
    """Handle the export-markdown subcommand.

    This command transcribes an audio file and emits a formatted Markdown transcript,
    alongside JSON metadata and segment files.

    Args:
        args: Parsed CLI arguments
        console_manager: Optional console manager for rich output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate input audio file
        try:
            audio_path = validate_audio_file(args.audio_path)
        except ValidationError:
            return 1

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Transcribe
        service = TranscriptionService()
        provider_name = args.provider if args.provider != "auto" else None

        logger.info(f"Transcribing {audio_path} using {args.provider} provider...")
        result = service.transcribe(audio_path, provider_name=provider_name, language=args.language)

        if not result:
            logger.error("Transcription failed")
            return 1

        # Create output directory structure
        safe_name = sanitize_dirname(audio_path.stem)
        base_dir = ensure_subpath(output_dir, Path(safe_name))
        base_dir.mkdir(parents=True, exist_ok=True)

        # Generate markdown
        formatter = MarkdownFormatter()
        source_info = {
            "source": str(audio_path),
            "processed_at": datetime.now().isoformat(),
            "provider": result.provider_name,
            "total_duration": result.duration,
        }

        # Create formatted transcript
        md_path = base_dir / "transcript.md"
        md_content = formatter.format_transcript(
            result,
            source_info,
            md_path,
            include_timestamps=args.include_timestamps,
            include_speakers=args.include_speakers,
            include_confidence=args.include_confidence,
            template=args.template,
        )

        # Save transcript
        formatter.save_transcript(md_content, md_path)
        logger.info(f"Markdown transcript saved to: {md_path}")

        # Save metadata
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

        # Save segments
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

        logger.info("Export completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Export markdown command failed: {e}")
        return 1

