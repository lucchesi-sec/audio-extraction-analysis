"""Shared CLI utilities and argument parser."""
from __future__ import annotations

import argparse
import logging

from ..config.factory import ConfigFactory, ConfigType

__version__ = "1.0.0+emergency"

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration based on verbosity level.

    Args:
        verbose: If True, set to DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

    # Set specific loggers
    logging.getLogger("src").setLevel(level)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="audio-extraction-analysis",
        description="Audio extraction and transcription analysis tool with multiple providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Extract audio from video
  audio-extraction-analysis extract video.mp4 --quality speech
  
  # Transcribe audio file with auto provider selection
  audio-extraction-analysis transcribe audio.mp3 --language en
  
  # Transcribe with specific provider
  audio-extraction-analysis transcribe audio.mp3 --provider deepgram
  audio-extraction-analysis transcribe audio.mp3 --provider elevenlabs
  
  # Full pipeline: video to transcript
  audio-extraction-analysis process video.mp4 --output-dir ./results
  
  # With specific provider and verbose logging
  audio-extraction-analysis process video.mp4 --provider deepgram --verbose

Quality presets:
  high       - 320k bitrate, best for archival
  standard   - Variable bitrate, good balance
  speech     - Mono, normalized, best for transcription (default)
  compressed - 128k bitrate, smaller files

Transcription providers:
  deepgram   - Full-featured with speaker diarization, topics, intents, sentiment
  elevenlabs - Basic transcription with timestamps
  whisper    - Local OpenAI Whisper processing (no API key needed)
  auto       - Automatically select best available provider (default)

For more information, see: https://github.com/lucchesi-sec/audio-extraction-analysis
        """,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Emit machine-readable JSON events to stderr/stdout",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract audio from video files",
        description="Extract audio from video files using FFmpeg with quality presets",
    )
    extract_parser.add_argument("input_file", help="Input video file path")
    extract_parser.add_argument(
        "--output", "-o", help="Output audio file path (default: <input>.mp3)"
    )
    extract_parser.add_argument(
        "--quality",
        "-q",
        choices=["high", "standard", "speech", "compressed"],
        default="speech",
        help="Audio quality preset (default: speech)",
    )

    # Transcribe subcommand
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio files using multiple providers",
        description="Transcribe audio with provider selection (Deepgram Nova 3, ElevenLabs)",
    )
    transcribe_parser.add_argument("audio_file", help="Input audio file path")
    transcribe_parser.add_argument(
        "--output", "-o", help="Output transcript file path (default: <audio>_transcript.txt)"
    )
    transcribe_parser.add_argument(
        "--language", "-l", default="en", help="Language code for transcription (default: en)"
    )
    transcribe_parser.add_argument(
        "--provider",
        "-p",
        choices=["deepgram", "elevenlabs", "whisper", "auto"],
        default="auto",
        help="Transcription provider to use (default: auto)",
    )
    # Markdown export options for transcribe
    transcribe_parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Also export a formatted Markdown transcript into ./output/<name>/",
    )
    transcribe_parser.add_argument(
        "--md-template",
        dest="md_template",
        choices=["default", "minimal", "detailed"],
        default="default",
        help="Markdown template to use",
    )
    transcribe_parser.add_argument(
        "--md-no-timestamps",
        dest="md_include_timestamps",
        action="store_false",
        help="Exclude timestamps in Markdown output",
    )
    transcribe_parser.add_argument(
        "--md-no-speakers",
        dest="md_include_speakers",
        action="store_false",
        help="Exclude speaker labels in Markdown output",
    )
    transcribe_parser.add_argument(
        "--md-confidence",
        dest="md_include_confidence",
        action="store_true",
        help="Include confidence field in Markdown output",
    )

    # Process subcommand (extract + transcribe)
    process_parser = subparsers.add_parser(
        "process",
        help="Full pipeline: extract audio and transcribe",
        description="Complete video-to-transcript pipeline with audio extraction and transcription",
    )
    process_parser.add_argument("video_file", help="Input video file path")
    process_parser.add_argument(
        "--output-dir", "-o", help="Output directory for results (default: ./output)"
    )
    process_parser.add_argument(
        "--quality",
        "-q",
        choices=["high", "standard", "speech", "compressed"],
        default="speech",
        help="Audio quality preset (default: speech)",
    )
    process_parser.add_argument(
        "--language", "-l", default="en", help="Language code for transcription (default: en)"
    )
    process_parser.add_argument(
        "--provider",
        "-p",
        choices=["deepgram", "elevenlabs", "whisper", "auto"],
        default="auto",
        help="Transcription provider to use (default: auto)",
    )
    process_parser.add_argument(
        "--analysis-style",
        "-a",
        choices=["concise", "full"],
        default="concise",
        help=(
            "Analysis output style: 'concise' for single comprehensive file, "
            "'full' for 5 detailed files (default: concise)"
        ),
    )
    # Markdown export options for process
    process_parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Also export a formatted Markdown transcript into <output_dir>/<name>/",
    )
    process_parser.add_argument(
        "--md-template",
        dest="md_template",
        choices=["default", "minimal", "detailed"],
        default="default",
        help="Markdown template to use",
    )
    process_parser.add_argument(
        "--md-no-timestamps",
        dest="md_include_timestamps",
        action="store_false",
        help="Exclude timestamps in Markdown output",
    )
    process_parser.add_argument(
        "--md-no-speakers",
        dest="md_include_speakers",
        action="store_false",
        help="Exclude speaker labels in Markdown output",
    )
    process_parser.add_argument(
        "--md-confidence",
        dest="md_include_confidence",
        action="store_true",
        help="Include confidence field in Markdown output",
    )

    # Export markdown subcommand
    export_md_parser = subparsers.add_parser(
        "export-markdown",
        help="Transcribe audio and export formatted Markdown transcript",
        description=(
            "Generate professionally formatted Markdown transcripts "
            "with timestamps, speaker labels, and metadata."
        ),
    )
    export_md_parser.add_argument("audio_path", help="Path to audio file")
    export_md_parser.add_argument(
        "--output-dir", "-o", default="./output", help="Output directory (default: ./output)"
    )
    export_md_parser.add_argument(
        "--provider",
        "-p",
        choices=["deepgram", "elevenlabs", "whisper", "auto"],
        default="auto",
        help="Transcription provider to use (default: auto)",
    )
    export_md_parser.add_argument(
        "--language",
        "-l",
        default="en",
        help="Language code (default from config)",
    )
    # Paired flags for booleans in argparse
    export_md_parser.add_argument(
        "--timestamps",
        dest="include_timestamps",
        action="store_true",
        help="Include timestamps in transcript",
    )
    export_md_parser.add_argument(
        "--no-timestamps",
        dest="include_timestamps",
        action="store_false",
        help="Exclude timestamps in transcript",
    )
    export_md_parser.set_defaults(include_timestamps=True)
    export_md_parser.add_argument(
        "--speakers",
        dest="include_speakers",
        action="store_true",
        help="Include speaker labels",
    )
    export_md_parser.add_argument(
        "--no-speakers",
        dest="include_speakers",
        action="store_false",
        help="Exclude speaker labels",
    )
    export_md_parser.set_defaults(include_speakers=True)
    export_md_parser.add_argument(
        "--confidence",
        action="store_true",
        dest="include_confidence",
        help="Include confidence indicators when available",
    )
    export_md_parser.add_argument(
        "--template",
        default="default",
        choices=["default", "minimal", "detailed"],
        help="Markdown template to use",
    )

    return parser
