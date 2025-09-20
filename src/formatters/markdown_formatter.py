from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config.config import Config
from ..models.transcription import (
    TranscriptionResult,
    TranscriptionUtterance,
)
from ..utils.paths import sanitize_dirname as util_sanitize_dirname
from .templates import TEMPLATES


class MarkdownFormattingError(Exception):
    """Raised when markdown formatting fails."""


class TemplateNotFoundError(Exception):
    """Raised when specified template doesn't exist."""


class MarkdownFormatter:
    """Formats transcription results as professionally structured markdown."""

    def __init__(self, config: Config | None = None):
        # Config is a class with class variables; keep reference for defaults
        self.config = config or Config

    def format_transcript(
        self,
        result: TranscriptionResult,
        source_info: dict[str, Any],
        output_path: Path,
        include_timestamps: bool = True,
        include_speakers: bool = True,
        include_confidence: bool = False,
        template: str = "default",
    ) -> str:
        """Generate markdown-formatted transcript with metadata."""
        try:
            tpl = TEMPLATES.get(template)
            if tpl is None:
                raise TemplateNotFoundError(f"Template '{template}' not found")

            header = self._format_header(result, source_info, tpl)
            segments_md = self._format_segments(
                result.utterances or [],
                include_timestamps=include_timestamps,
                include_speakers=include_speakers,
                include_confidence=include_confidence,
                template=tpl,
            )

            # Fallback if no utterances provided: dump full transcript
            if not segments_md.strip():
                # Use minimal template segment rendering
                text_block = result.transcript.strip() if result.transcript else ""
                if text_block:
                    seg_tpl = str(tpl.get("segment", "{text}\n\n"))
                    segments_md = seg_tpl.format(
                        timestamp=(
                            "00:00:00" if include_timestamps and tpl.get("timestamp_format") else ""
                        ),
                        speaker_prefix=(
                            ""
                            if not include_speakers
                            else (str(tpl.get("speaker_prefix", ""))).format(speaker="")
                        ),
                        text=text_block,
                        confidence="" if not include_confidence else "",
                    )

            return header + segments_md
        except TemplateNotFoundError:
            raise
        except Exception as e:
            raise MarkdownFormattingError(f"Failed to format transcript: {e}") from e

    def _format_header(
        self, result: TranscriptionResult, source_info: dict[str, Any], tpl: dict[str, Any]
    ) -> str:
        """Generate markdown header with metadata."""
        header_tpl = str(tpl.get("header", "# Transcript\n\n"))

        title = Path(source_info.get("source", result.audio_file or "transcript")).stem
        duration_seconds = float(
            source_info.get("total_duration", getattr(result, "duration", 0.0) or 0.0)
        )
        duration = self._format_timestamp(duration_seconds)
        processed_at = source_info.get("processed_at") or datetime.now().isoformat()
        provider = source_info.get("provider") or result.provider_name
        segment_count = len(result.utterances or [])
        avg_confidence = source_info.get("avg_confidence", "N/A")

        return header_tpl.format(
            title=title,
            source=source_info.get("source", result.audio_file),
            duration=duration,
            processed_at=processed_at,
            provider=provider,
            segment_count=segment_count,
            avg_confidence=avg_confidence,
        )

    def _format_segments(
        self,
        segments: list[TranscriptionUtterance],
        include_timestamps: bool,
        include_speakers: bool,
        include_confidence: bool,
        template: dict[str, Any],
    ) -> str:
        """Format transcript segments with optional metadata."""
        if not segments:
            return ""

        seg_tpl = str(template.get("segment", "{text}\n\n"))
        sp_prefix_tpl = str(template.get("speaker_prefix", ""))
        use_ts = include_timestamps and bool(template.get("timestamp_format"))

        parts: list[str] = []
        for seg in segments:
            timestamp = self._format_timestamp(seg.start) if use_ts else ""
            speaker_prefix = (
                sp_prefix_tpl.format(speaker=f"Speaker {seg.speaker}") if include_speakers else ""
            )
            text = getattr(seg, "text", None) or getattr(seg, "transcript", "")
            confidence_str = ""
            if include_confidence:
                # Our model doesn't carry per-utterance confidence; mark as N/A
                confidence_str = "confidence: N/A"

            parts.append(
                seg_tpl.format(
                    timestamp=timestamp,
                    speaker_prefix=speaker_prefix,
                    text=text,
                    confidence=confidence_str,
                )
            )

        return "".join(parts)

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        seconds = max(0.0, float(seconds or 0.0))
        td = timedelta(seconds=int(seconds))
        # Ensure HH:MM:SS
        total_seconds = int(td.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def save_transcript(self, content: str, output_path: Path) -> None:
        """Save formatted transcript to file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encoding = getattr(self.config, "markdown_output_encoding", "utf-8")
            with open(output_path, "w", encoding=encoding) as f:
                f.write(content)
        except (OSError, PermissionError) as e:
            raise MarkdownFormattingError(
                f"Failed to write transcript to {output_path}: {e}"
            ) from e

    @staticmethod
    def sanitize_dirname(name: str) -> str:
        """Delegate to shared util sanitizer for consistency."""
        return util_sanitize_dirname(name)
