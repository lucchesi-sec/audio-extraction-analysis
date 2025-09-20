"""Audio preprocessing utilities for Parakeet transcription provider."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Audio processing libraries
try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio processing libraries (librosa, soundfile) not available")


class ParakeetAudioError(Exception):
    """Raised when audio processing fails."""

    pass


class AudioPreprocessor:
    """Handles audio preprocessing for Parakeet models."""

    # Parakeet models typically expect 16kHz audio
    TARGET_SAMPLE_RATE = 16000

    @classmethod
    def preprocess_audio(cls, audio_path: Path) -> Tuple[Optional[Path], Optional[float]]:
        """Preprocess audio file for Parakeet transcription.

        Converts audio to 16kHz mono WAV format if needed.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (processed_audio_path, duration_seconds)
            Returns (None, None) if preprocessing fails
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.error("Audio processing libraries not available")
            return None, None

        try:
            # Load audio
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=None,
                mono=True,  # Preserve original sample rate initially  # Convert to mono
            )

            # Calculate duration
            duration = len(audio_data) / sample_rate

            # Resample if needed
            if sample_rate != cls.TARGET_SAMPLE_RATE:
                logger.info(f"Resampling audio from {sample_rate}Hz to {cls.TARGET_SAMPLE_RATE}Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=cls.TARGET_SAMPLE_RATE
                )
                sample_rate = cls.TARGET_SAMPLE_RATE

            # Check if we need to save preprocessed audio
            needs_preprocessing = (
                audio_path.suffix.lower() != ".wav" or sample_rate != cls.TARGET_SAMPLE_RATE
            )

            if needs_preprocessing:
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=audio_path.parent
                ) as tmp_file:
                    output_path = Path(tmp_file.name)

                # Save preprocessed audio
                sf.write(
                    str(output_path), audio_data, cls.TARGET_SAMPLE_RATE, subtype="PCM_16"
                )  # 16-bit PCM

                logger.info(f"Preprocessed audio saved to {output_path}")
                return output_path, duration
            else:
                # Audio is already in correct format
                return audio_path, duration

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise ParakeetAudioError(f"Failed to preprocess audio: {e}")

    @classmethod
    def validate_audio_file(cls, audio_path: Path) -> bool:
        """Validate that audio file can be processed.

        Args:
            audio_path: Path to audio file

        Returns:
            True if file is valid and can be processed
        """
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False

        if not audio_path.is_file():
            logger.error(f"Path is not a file: {audio_path}")
            return False

        # Check file size (max 2GB)
        max_size = 2 * 1024 * 1024 * 1024
        if audio_path.stat().st_size > max_size:
            logger.error(f"Audio file too large: {audio_path.stat().st_size} bytes")
            return False

        # Check if we can read the file
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Cannot validate audio format without librosa/soundfile")
            # Optimistically return True - actual processing will fail if invalid
            return True

        try:
            # Try to get audio info
            info = sf.info(str(audio_path))
            logger.debug(
                f"Audio file info: duration={info.duration}s, "
                f"samplerate={info.samplerate}Hz, channels={info.channels}"
            )
            return True
        except Exception as e:
            logger.error(f"Invalid audio file {audio_path}: {e}")
            return False

    @classmethod
    def get_audio_duration(cls, audio_path: Path) -> Optional[float]:
        """Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds, or None if cannot be determined
        """
        if not AUDIO_LIBS_AVAILABLE:
            return None

        try:
            info = sf.info(str(audio_path))
            return info.duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return None

    @classmethod
    def cleanup_temp_file(cls, temp_path: Optional[Path]) -> None:
        """Clean up temporary audio file.

        Args:
            temp_path: Path to temporary file to delete
        """
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
