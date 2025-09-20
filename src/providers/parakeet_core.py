"""NVIDIA Parakeet STT transcription service - core implementation."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.transcription import TranscriptionResult, TranscriptionUtterance
from ..utils.retry import RetryConfig
from .base import BaseTranscriptionProvider, CircuitBreakerConfig
from .parakeet_audio import AudioPreprocessor
from .parakeet_cache import NEMO_AVAILABLE, ParakeetModelCache

# Import split modules
from .parakeet_gpu import TORCH_AVAILABLE, GPUManager

logger = logging.getLogger(__name__)

# Parakeet model definitions
PARAKEET_MODELS = {
    "stt_en_conformer_ctc_large": {
        "type": "ctc",
        "accuracy": "high",
        "speed": "fast",
        "memory": "4GB",
        "languages": ["en"],
    },
    "stt_en_conformer_transducer_large": {
        "type": "rnnt",
        "accuracy": "highest",
        "speed": "medium",
        "memory": "6GB",
        "languages": ["en"],
    },
    "stt_en_fastconformer_ctc_large": {
        "type": "ctc",
        "accuracy": "medium",
        "speed": "fastest",
        "memory": "2GB",
        "languages": ["en"],
    },
}


# Custom exception for core errors
class ParakeetError(Exception):
    """Base exception for Parakeet-specific errors."""

    pass


class ParakeetMetrics:
    """Tracks metrics for Parakeet transcriptions."""

    def __init__(self):
        self.total_transcriptions = 0
        self.total_duration = 0.0
        self.total_processing_time = 0.0

    def log_transcription(self, duration: float, audio_length: float) -> None:
        """Log a transcription event.

        Args:
            duration: Processing time in seconds
            audio_length: Audio length in seconds
        """
        self.total_transcriptions += 1
        self.total_duration += audio_length
        self.total_processing_time += duration

    def get_rtf(self) -> float:
        """Get real-time factor (processing time / audio time).

        Returns:
            RTF value, or 0.0 if no data
        """
        if self.total_duration > 0:
            return self.total_processing_time / self.total_duration
        return 0.0


class ParakeetTranscriber(BaseTranscriptionProvider):
    """NVIDIA Parakeet STT transcription service with CTC/RNN-T model support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the Parakeet transcriber.

        Args:
            api_key: Optional API key (not used for local Parakeet)
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration
        """
        super().__init__(api_key, circuit_config, retry_config)
        self.gpu_manager = GPUManager()
        self.model_cache = ParakeetModelCache()
        self.audio_preprocessor = AudioPreprocessor()
        self.metrics = ParakeetMetrics()

        # Configuration from environment
        self.model_name = os.getenv("PARAKEET_MODEL", "stt_en_conformer_ctc_large")
        self.batch_size = int(os.getenv("PARAKEET_BATCH_SIZE", "8"))
        self.beam_size = int(os.getenv("PARAKEET_BEAM_SIZE", "10"))
        self.use_fp16 = os.getenv("PARAKEET_USE_FP16", "true").lower() == "true"
        self.chunk_length = int(os.getenv("PARAKEET_CHUNK_LENGTH", "30"))
        self.model_cache_dir = os.getenv("PARAKEET_MODEL_CACHE_DIR", "~/.cache/parakeet")

    def validate_configuration(self) -> bool:
        """Validate that Parakeet is properly configured.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check dependencies
            if not NEMO_AVAILABLE:
                logger.error(
                    "NeMo toolkit not installed. Install with: pip install nemo-toolkit[asr]"
                )
                return False

            if not TORCH_AVAILABLE:
                logger.error("PyTorch not installed")
                return False

            # Validate model name
            if not self.model_name or not isinstance(self.model_name, str):
                logger.error("Invalid model name format")
                return False

            if len(self.model_name) > 200:
                logger.error(f"Model name too long: {len(self.model_name)}")
                return False

            if self.model_name not in PARAKEET_MODELS:
                logger.error(f"Unsupported Parakeet model: {self.model_name}")
                return False

            # Validate configuration parameters
            if self.batch_size <= 0 or self.batch_size > 32:
                logger.error(f"Invalid batch size: {self.batch_size}")
                return False

            if self.beam_size <= 0 or self.beam_size > 50:
                logger.error(f"Invalid beam size: {self.beam_size}")
                return False

            if self.chunk_length <= 0 or self.chunk_length > 300:
                logger.error(f"Invalid chunk length: {self.chunk_length}")
                return False

            # Check device compatibility
            device = self.gpu_manager.device
            if device.startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    logger.error("CUDA not available but required by GPU manager")
                    return False

                # Test GPU functionality
                try:
                    torch.cuda.empty_cache()
                    test_tensor = torch.randn(10, 10, device="cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"GPU test failed: {e}")
                    return False

            # Validate model cache directory
            try:
                cache_dir = Path(self.model_cache_dir).expanduser()
                if not cache_dir.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                if not cache_dir.is_dir() or not os.access(cache_dir, os.W_OK):
                    logger.error(f"Model cache directory not writable: {cache_dir}")
                    return False
            except Exception as e:
                logger.error(f"Model cache directory validation failed: {e}")
                return False

            # Check memory requirements
            estimated_size = self.model_cache._estimate_model_size(self.model_name)
            if not self.gpu_manager.can_allocate_model(estimated_size):
                available = self.gpu_manager.get_available_memory()
                logger.error(
                    f"Insufficient memory for model {self.model_name}. "
                    f"Required: {estimated_size / 1024**2:.1f}MB, "
                    f"Available: {available / 1024**2 if available else 'Unknown'}MB"
                )
                return False

            logger.info(
                f"Parakeet configuration validated successfully: {self.model_name} on {device}"
            )
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """Get the name of this transcription provider.

        Returns:
            Human-readable name of the provider
        """
        model_type = PARAKEET_MODELS.get(self.model_name, {}).get("type", "unknown")
        return f"NVIDIA Parakeet ({model_type})"

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by Parakeet.

        Returns:
            List of supported feature names
        """
        return [
            "timestamps",
            "speaker_diarization",
            "language_detection",
            "punctuation_restoration",
            "local_processing",
            "offline_capable",
            "gpu_acceleration",
        ]

    async def _transcribe_impl(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Internal implementation of Parakeet transcription.

        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription

        Returns:
            TranscriptionResult object or None if failed
        """
        try:
            if not NEMO_AVAILABLE or not TORCH_AVAILABLE:
                logger.error("Required dependencies not available")
                return None

            # Validate audio file
            if not self.audio_preprocessor.validate_audio_file(audio_file_path):
                logger.error(f"Invalid audio file: {audio_file_path}")
                return None

            # Preprocess audio if needed
            processed_path, audio_duration = self.audio_preprocessor.preprocess_audio(
                audio_file_path
            )
            if processed_path is None:
                logger.error("Audio preprocessing failed")
                return None

            temp_file_created = processed_path != audio_file_path

            try:
                # Get model from cache
                model = await self.model_cache.get_model_async(self.model_name)
                if model is None:
                    logger.error("Failed to load model")
                    return None

                # Prepare transcription kwargs
                transcription_kwargs = {
                    "paths2audio_files": [str(processed_path)],
                    "batch_size": self.batch_size,
                    "return_hypotheses": False,
                    "verbose": False,
                }

                # Add beam search for RNN-T models
                if PARAKEET_MODELS.get(self.model_name, {}).get("type") == "rnnt":
                    transcription_kwargs["beam_size"] = self.beam_size

                # Run transcription
                start_time = time.time()
                transcription = await self._run_transcription(
                    model, transcription_kwargs, self.gpu_manager.device
                )
                processing_time = time.time() - start_time

                # Log metrics
                self.metrics.log_transcription(processing_time, audio_duration)

                # Parse result
                return self._parse_parakeet_result(
                    transcription, audio_file_path, processing_time, audio_duration
                )

            finally:
                # Clean up temporary file
                if temp_file_created:
                    self.audio_preprocessor.cleanup_temp_file(processed_path)

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            return None

    async def _run_transcription(
        self, model: Any, kwargs: Dict[str, Any], device: str
    ) -> List[str]:
        """Run transcription in thread pool to avoid blocking.

        Args:
            model: Loaded Parakeet model
            kwargs: Transcription arguments
            device: Device to run on

        Returns:
            List of transcription results
        """
        loop = asyncio.get_event_loop()

        def _transcribe():
            try:
                if TORCH_AVAILABLE and device.startswith("cuda") and self.use_fp16:
                    import torch

                    with torch.cuda.amp.autocast():
                        return model.transcribe(**kwargs)
                else:
                    return model.transcribe(**kwargs)
            except Exception as e:
                logger.error(f"Model transcribe call failed: {e}")
                raise

        return await loop.run_in_executor(None, _transcribe)

    def _parse_parakeet_result(
        self,
        parakeet_result: List[str],
        audio_file_path: Path,
        processing_time: float,
        audio_duration: float,
    ) -> TranscriptionResult:
        """Parse Parakeet result into TranscriptionResult format.

        Args:
            parakeet_result: Raw Parakeet transcription result
            audio_file_path: Path to the source audio file
            processing_time: Time taken for transcription
            audio_duration: Actual duration of the audio in seconds

        Returns:
            Formatted TranscriptionResult
        """
        try:
            # Join transcription results
            transcript = (
                " ".join(str(result) for result in parakeet_result if result)
                if parakeet_result
                else ""
            )
            transcript = transcript.strip()

            # Create utterances
            utterances = []
            if transcript:
                utterances.append(
                    TranscriptionUtterance(
                        speaker=1, start=0.0, end=audio_duration, text=transcript
                    )
                )

            # Calculate metrics
            word_count = len(transcript.split()) if transcript else 0
            rtf = self.metrics.get_rtf()

            # Create metadata
            metadata = {
                "parakeet_model": self.model_name,
                "model_type": PARAKEET_MODELS.get(self.model_name, {}).get("type", "unknown"),
                "device": self.gpu_manager.device,
                "use_fp16": self.use_fp16,
                "batch_size": self.batch_size,
                "beam_size": self.beam_size,
                "chunk_length": self.chunk_length,
                "processing_time_seconds": processing_time,
                "audio_duration_seconds": audio_duration,
                "rtf": rtf,
                "words_per_minute": (
                    (word_count / (audio_duration / 60)) if audio_duration > 0 else 0
                ),
                "transcription_confidence": 1.0,
                "language": PARAKEET_MODELS.get(self.model_name, {}).get("languages", ["en"])[0],
                "sample_rate": 16000,
                "channels": 1,
            }

            # Add GPU memory information if available
            if self.gpu_manager.device.startswith("cuda") and TORCH_AVAILABLE:
                try:
                    import torch

                    available_memory = self.gpu_manager.get_available_memory()
                    if available_memory is not None:
                        metadata["gpu_memory_available_mb"] = available_memory / 1024**2

                    allocated_memory = torch.cuda.memory_allocated()
                    metadata["gpu_memory_allocated_mb"] = allocated_memory / 1024**2
                except Exception:
                    pass

            # Create the result
            return TranscriptionResult(
                transcript=transcript,
                duration=audio_duration,
                generated_at=datetime.now(),
                audio_file=str(audio_file_path),
                provider_name=self.get_provider_name(),
                utterances=utterances,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse Parakeet result: {e}")
            # Return minimal result on parsing failure
            return TranscriptionResult(
                transcript="",
                duration=audio_duration,
                generated_at=datetime.now(),
                audio_file=str(audio_file_path),
                provider_name=self.get_provider_name(),
                utterances=[],
                metadata={
                    "error": f"Result parsing failed: {e}",
                    "processing_time_seconds": processing_time,
                    "audio_duration_seconds": audio_duration,
                },
            )

    async def health_check_async(self) -> Dict[str, Any]:
        """Perform health check for Parakeet provider.

        Returns:
            Dictionary containing health check results
        """
        start_time = time.time()

        try:
            if not NEMO_AVAILABLE:
                return {
                    "healthy": False,
                    "status": "dependencies_missing",
                    "response_time_ms": 0,
                    "details": {"provider": "parakeet", "error": "NeMo toolkit not installed"},
                }

            # Test model loading
            try:
                model = await self.model_cache.get_model_async(self.model_name)
                model_available = model is not None
            except Exception as e:
                logger.debug(f"Model loading test failed during health check: {e}")
                model_available = False

            health_status = {
                "healthy": model_available,
                "status": "ready" if model_available else "model_not_loaded",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {
                    "provider": "parakeet",
                    "model_loaded": model_available,
                    "model_name": self.model_name,
                    "device": self.gpu_manager.device,
                    "cuda_available": TORCH_AVAILABLE
                    and self.gpu_manager.device.startswith("cuda"),
                    "cache_stats": self.model_cache.get_cache_stats(),
                },
            }

            return health_status

        except Exception as e:
            return {
                "healthy": False,
                "status": "health_check_failed",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {
                    "provider": "parakeet",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }
