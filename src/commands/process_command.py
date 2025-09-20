"""Combined process command implementation (extract + transcribe)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from ..pipeline.audio_pipeline import AudioProcessingPipeline
from ..services.audio_extraction import AudioQuality
from ..ui.console import ConsoleManager
from .transcribe_command import export_markdown_transcript

logger = logging.getLogger(__name__)


def process_command(args: argparse.Namespace, console_manager: Optional[ConsoleManager] = None) -> int:
    """Handle the process subcommand (extract + transcribe).

    Args:
        args: Command line arguments
        console_manager: Optional console manager for rich output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        input_path = Path(args.video_file)
        if not input_path.exists():
            logger.error(f"Video file not found: {input_path}")
            return 1

        # Setup output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("output")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse quality preset
        quality_map = {
            "high": AudioQuality.HIGH,
            "standard": AudioQuality.STANDARD,
            "speech": AudioQuality.SPEECH,
            "compressed": AudioQuality.COMPRESSED,
        }

        quality = quality_map.get(args.quality, AudioQuality.SPEECH)

        logger.info(
            f"Processing video {input_path} (quality: {quality.value}, provider: {args.provider})"
        )

        # Use async pipeline with progress bars (prefer class from src.cli for legacy patching)
        from .. import cli as cli_module

        pipeline_cls = getattr(cli_module, "AudioProcessingPipeline", AudioProcessingPipeline)
        pipeline = pipeline_cls(console_manager=console_manager)
        try:
            if hasattr(pipeline, "process_video"):
                # Legacy synchronous pipeline path expected by older tests
                result = pipeline.process_video(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    quality=quality.value if hasattr(quality, "value") else quality,
                    provider=args.provider,
                    language=args.language,
                )
                pipeline_result = {"success": result is not None, "transcript": result}
            else:
                # Use async method with progress bars
                pipeline_result = asyncio.run(
                    pipeline.process_file(
                        input_path=str(input_path),
                        output_dir=str(output_dir),
                        quality=quality,  # Pass the enum object
                        language=args.language,
                        provider=args.provider,
                        analysis_style=args.analysis_style,
                    )
                )

                # Extract the transcription result from pipeline results
                if pipeline_result.get("success", False):
                    result = pipeline_result.get("transcript")  # TranscriptionResult object
                else:
                    result = None
            if not pipeline_result.get("success", False):
                errors = pipeline_result.get("errors", ["Unknown error"])
                logger.error(f"Pipeline processing failed: {', '.join(errors)}")
                # Targeted diagnostics: dump stage results and context
                import os
                if os.getenv("AUDIO_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
                    diag = {
                        "stage_results": pipeline_result.get("stage_results"),
                        "stages_completed": pipeline_result.get("stages_completed"),
                        "files_created": pipeline_result.get("files_created"),
                        "audio_path": pipeline_result.get("audio_path"),
                        "debug_path": pipeline_result.get("debug_path"),
                    }
                    try:
                        logger.error("Pipeline diagnostics: %s", json.dumps(diag, default=str))
                    except Exception:
                        logger.error(f"Pipeline diagnostics (raw): {diag}")
        except Exception as e:
            logger.error(f"Process command failed during pipeline: {e}")
            return 1

        if result:
            logger.info("Processing completed successfully!")
            logger.info(f"Results saved to: {output_dir}")

            # Optional Markdown export
            if getattr(args, "export_markdown", False):
                # Update args to use output_dir for markdown output
                args.markdown_output_dir = str(output_dir)
                export_markdown_transcript(args, input_path, result)

            return 0
        else:
            logger.error("Processing failed")
            return 1

    except Exception as e:
        logger.error(f"Process command failed: {e}")
        return 1
