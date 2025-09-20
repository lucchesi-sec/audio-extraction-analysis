"""OpenAI Whisper transcription service with local/cloud support."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.config import Config
from ..models.transcription import TranscriptionResult, TranscriptionUtterance
from ..utils.retry import RetryConfig
from .base import BaseTranscriptionProvider, CircuitBreakerConfig

logger = logging.getLogger(__name__)

# Lazy dependency resolution to avoid import-time failures in environments
# where Whisper/torch are not installed or incompatible.
PROVIDER_AVAILABLE = None  # resolved on first use
whisper = None
torch = None
get_writer = None

def _ensure_whisper_available() -> bool:
    global PROVIDER_AVAILABLE, whisper, torch, get_writer
    if PROVIDER_AVAILABLE is not None:
        return PROVIDER_AVAILABLE
    try:
        import torch as _torch  # type: ignore
        import whisper as _whisper  # type: ignore
        try:
            from whisper.utils import get_writer as _get_writer  # type: ignore
        except Exception:
            _get_writer = None
        torch = _torch
        whisper = _whisper
        get_writer = _get_writer
        PROVIDER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Whisper provider dependencies not installed: {e}")
        PROVIDER_AVAILABLE = False
    return PROVIDER_AVAILABLE


class WhisperTranscriber(BaseTranscriptionProvider):
    """OpenAI Whisper transcription service with local processing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the Whisper transcriber.

        Args:
            api_key: Optional API key (not used for local Whisper)
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration
        """
        super().__init__(api_key, circuit_config, retry_config)
        self.model = None
        self.model_name = Config.WHISPER_MODEL or "base"
        self.device = (
            Config.WHISPER_DEVICE or "cuda" if torch and torch.cuda.is_available() else "cpu"
        )
        self.compute_type = Config.WHISPER_COMPUTE_TYPE or "float16"

    def validate_configuration(self) -> bool:
        """Validate that Whisper is properly configured.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not _ensure_whisper_available():
            logger.error(
                "Whisper dependencies not installed. Install with: pip install openai-whisper torch"
            )
            return False

        # Check if device is available
        if self.device == "cuda" and torch and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        return True

    def get_provider_name(self) -> str:
        """Get the name of this transcription provider.

        Returns:
            Human-readable name of the provider
        """
        return f"OpenAI Whisper ({self.model_name})"

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by Whisper.

        Returns:
            List of supported feature names
        """
        return [
            "timestamps",
            "word_timestamps",
            "language_detection",
            "vad_filter",  # Voice Activity Detection
            "local_processing",
            "offline_capable",
        ]

    async def _load_model(self) -> None:
        """Load the Whisper model asynchronously."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            try:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, whisper.load_model, self.model_name, self.device
                )
                logger.info(f"Whisper model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

    async def _transcribe_impl(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Internal implementation of Whisper transcription.

        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription

        Returns:
            TranscriptionResult object
        """
        if not _ensure_whisper_available():
            raise ImportError("Whisper dependencies not installed")

        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        await self._load_model()

        # Prepare transcription options
        options = {
            "language": language if language != "auto" else None,
            "task": "transcribe",
            "fp16": self.compute_type == "float16",
            "verbose": False,
        }

        logger.info(f"Transcribing with Whisper: {audio_file_path.name}")
        start_time = time.time()

        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.model.transcribe, str(audio_file_path), **options
            )

            processing_time = time.time() - start_time
            logger.info(f"Whisper transcription completed in {processing_time:.2f}s")

            return self._parse_whisper_result(result, audio_file_path, processing_time)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def _parse_whisper_result(
        self, whisper_result: Dict, audio_file_path: Path, processing_time: float
    ) -> TranscriptionResult:
        """Parse Whisper result into TranscriptionResult format.

        Args:
            whisper_result: Raw Whisper transcription result
            audio_file_path: Path to the source audio file
            processing_time: Time taken for transcription

        Returns:
            Formatted TranscriptionResult
        """
        utterances = []

        for segment in whisper_result.get("segments", []):
            utterance = TranscriptionUtterance(
                speaker=1,  # Whisper doesn't support speaker diarization, default to speaker 1
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
            )
            utterances.append(utterance)

        # Calculate total duration from audio file or segments
        total_duration = max(utterance.end for utterance in utterances) if utterances else 0

        # Create metadata dict
        metadata = {
            "whisper_model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "has_words": any(
                segment.get("words") for segment in whisper_result.get("segments", [])
            ),
            "chapters": self._generate_chapters(utterances),
            "language": whisper_result.get("language", "en"),
            "processing_time_seconds": processing_time,
        }

        result = TranscriptionResult(
            transcript=" ".join(
                segment["text"].strip() for segment in whisper_result.get("segments", [])
            ),
            duration=total_duration,
            generated_at=datetime.now(),
            audio_file=str(audio_file_path),
            provider_name=self.get_provider_name(),
            metadata=metadata,
        )

        return result

    def _extract_words(self, segment: Dict) -> Optional[List[Dict]]:
        """Extract word-level timestamps from Whisper segment.

        Args:
            segment: Whisper segment with word information

        Returns:
            List of word dictionaries or None
        """
        if not segment.get("words"):
            return None

        words = []
        for word_info in segment["words"]:
            words.append(
                {
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "confidence": word_info.get("probability", 0.0),
                }
            )

        return words

    def _generate_chapters(self, utterances: List[TranscriptionUtterance]) -> List[Dict]:
        """Generate simple chapters based on time intervals.

        Args:
            utterances: List of transcription utterances

        Returns:
            List of chapter dictionaries
        """
        if not utterances:
            return []

        # Simple chapter generation: every 5 minutes
        chapters = []
        total_duration = max(utt.end for utt in utterances)

        for i in range(0, int(total_duration) + 300, 300):  # Every 5 minutes
            if i <= total_duration:
                chapters.append({"start_time": i, "end_time": min(i + 300, total_duration)})

        return chapters

    async def health_check_async(self) -> Dict[str, Any]:
        """Perform health check for Whisper provider.

        Returns:
            Dictionary containing health check results
        """
        start_time = time.time()

        try:
            if not PROVIDER_AVAILABLE:
                return {
                    "healthy": False,
                    "status": "dependencies_missing",
                    "response_time_ms": 0,
                    "details": {
                        "provider": "whisper",
                        "error": "Whisper dependencies not installed",
                    },
                }

            # Test model loading if not already loaded
            if self.model is None:
                await self._load_model()

            health_status = {
                "healthy": self.model is not None,
                "status": "ready" if self.model else "model_not_loaded",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {
                    "provider": "whisper",
                    "model_loaded": self.model is not None,
                    "model_name": self.model_name,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "cuda_available": torch.cuda.is_available() if torch else False,
                },
            }

            return health_status

        except Exception as e:
            return {
                "healthy": False,
                "status": "health_check_failed",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {"provider": "whisper", "error": str(e), "error_type": type(e).__name__},
            }

    def get_model_size_info(self) -> Dict[str, Any]:
        """Get information about Whisper model sizes and requirements.

        Returns:
            Dictionary with model size information
        """
        model_sizes = {
            "tiny": {"params": "39M", "disk": "75MB", "ram": "~1GB", "vram": "~1GB"},
            "base": {"params": "74M", "disk": "142MB", "ram": "~1GB", "vram": "~1GB"},
            "small": {"params": "244M", "disk": "461MB", "ram": "~2GB", "vram": "~2GB"},
            "medium": {"params": "769M", "disk": "1.5GB", "ram": "~5GB", "vram": "~5GB"},
            "large": {"params": "1.5B", "disk": "2.9GB", "ram": "~10GB", "vram": "~10GB"},
            "large-v2": {"params": "1.5B", "disk": "2.9GB", "ram": "~10GB", "vram": "~10GB"},
            "large-v3": {"params": "1.5B", "disk": "2.9GB", "ram": "~10GB", "vram": "~10GB"},
        }

        return {
            "current_model": self.model_name,
            "current_info": model_sizes.get(
                self.model_name, {"params": "unknown", "disk": "unknown"}
            ),
            "available_models": list(model_sizes.keys()),
            "model_sizes": model_sizes,
        }
