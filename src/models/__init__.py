"""Data models for the audio extraction analysis system."""

from .transcription import (
    TranscriptionChapter,
    TranscriptionResult,
    TranscriptionSpeaker,
    TranscriptionUtterance,
)

__all__ = [
    "TranscriptionChapter",
    "TranscriptionResult",
    "TranscriptionSpeaker",
    "TranscriptionUtterance",
]
