"""Shared FFmpeg helpers for sync and async extractors.

This module centralizes command construction and common behaviors to
reduce duplication between `audio_extraction.py` and `audio_extraction_async.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


def build_base_cmd(input_path: Path) -> List[str]:
    """Base ffmpeg command for extraction with overwrite enabled."""
    return ["ffmpeg", "-i", str(input_path), "-y"]


def build_extract_commands(
    input_path: Path, output_path: Path, quality: str
) -> Tuple[List[List[str]], Optional[Path]]:
    """Build one or two ffmpeg commands depending on quality preset.

    For `SPEECH`, returns a two-step pipeline: extract -> normalize/mono.
    For other presets, returns a single extraction command.

    Returns:
        (commands, temp_path) where temp_path is used for SPEECH cleanup.
    """
    base = build_base_cmd(input_path)

    if quality == "high":
        extract = [*base, "-b:a", "320k", "-map", "a", str(output_path)]
        return [extract], None

    if quality == "standard":
        extract = [*base, "-q:a", "0", "-map", "a", str(output_path)]
        return [extract], None

    if quality == "compressed":
        extract = [*base, "-b:a", "128k", "-map", "a", str(output_path)]
        return [extract], None

    # Default to SPEECH behavior
    temp_path = output_path.with_suffix(".temp.mp3")
    extract = [*base, "-q:a", "0", "-map", "a", str(temp_path)]
    normalize = [
        "ffmpeg",
        "-i",
        str(temp_path),
        "-y",
        "-ac",
        "1",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]
    return [extract, normalize], temp_path
