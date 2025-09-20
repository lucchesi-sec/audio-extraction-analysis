"""Deepgram Nova 3 transcription service with comprehensive features."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.config import Config
from ..models.transcription import (
    TranscriptionChapter,
    TranscriptionResult,
    TranscriptionSpeaker,
    TranscriptionUtterance,
)
from ..utils.retry import RetryConfig
from .deepgram_utils import detect_mimetype as _dg_detect_mimetype, build_prerecorded_options
from .base import BaseTranscriptionProvider, CircuitBreakerConfig
from .provider_utils import ProviderInitializer
from ..utils.file_validation import safe_validate_audio_file

logger = logging.getLogger(__name__)

# Check for Deepgram SDK availability
try:
    from deepgram import DeepgramClient, PrerecordedOptions

    PROVIDER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Deepgram provider dependencies not installed: {e}")
    PROVIDER_AVAILABLE = False
    # Create placeholder classes to prevent import errors
    DeepgramClient = None
    PrerecordedOptions = None
    FileSource = None


class DeepgramTranscriber(BaseTranscriptionProvider):
    """Deepgram Nova 3 transcription service with full feature support."""

    # ---------------------- Internal helpers (extracted) ----------------------
    def _create_client(self):
        """Create and return a Deepgram client configured with sane defaults.

        Returns:
            Deepgram client instance
        """
        # Import Deepgram SDK lazily to avoid import-time failures when optional
        from deepgram import ClientOptionsFromEnv, DeepgramClient  # type: ignore

        # 10 minute timeout (large files can take time)
        config = ClientOptionsFromEnv(options={"timeout": 600})
        return DeepgramClient(self.api_key, config=config)

    def _build_options(self, language: str):
        """Build PrerecordedOptions for Nova-3 with all features enabled.

        Args:
            language: Preferred language code
        """
        return build_prerecorded_options(language)

    def _detect_mimetype(self, path: Path) -> str:
        """Best-effort mimetype detection based on file extension.

        Defaults to audio/mp3 if unknown to keep behavior consistent with prior code.
        """
        return _dg_detect_mimetype(path)

    def _read_audio_file(self, audio_file_path: Path) -> bytes:
        """Read audio file into memory safely.

        Raises:
            OSError/PermissionError/FileNotFoundError propagated to caller
        """
        with open(audio_file_path, "rb") as audio_file:
            return audio_file.read()

    def _submit_transcription_job(self, client, buffer: bytes, mimetype: str, options) -> Any:
        """Submit the prerecorded transcription request.

        Args:
            client: Deepgram client
            buffer: Audio bytes
            mimetype: MIME type for audio
            options: PrerecordedOptions
        """
        # DG SDK: deepgram.listen.prerecorded.v("1").transcribe_file(...)
        return client.listen.prerecorded.v("1").transcribe_file(
            source={"buffer": buffer, "mimetype": mimetype}, options=options
        )

    def _parse_response(
        self, response: Any, audio_file_path: Path, language: str
    ) -> TranscriptionResult:
        """Parse Deepgram response into TranscriptionResult with all features.

        Args:
            response: Deepgram SDK response object
            audio_file_path: Source audio path
            language: Requested language
        """
        # Extract transcript and duration (required fields)
        transcript = response.results.channels[0].alternatives[0].transcript
        duration = response.metadata.duration

        result = TranscriptionResult(
            transcript=transcript,
            duration=duration,
            generated_at=datetime.now(),
            audio_file=str(audio_file_path),
            provider_name=self.get_provider_name(),
            provider_features=self.get_supported_features(),
        )

        # Summary
        if hasattr(response.results, "summary") and response.results.summary:
            result.summary = response.results.summary.short

        # Topics -> Chapters + topic counts
        if hasattr(response.results, "topics") and response.results.topics:
            for topic_segment in response.results.topics.segments:
                chapter = TranscriptionChapter(
                    start_time=getattr(topic_segment, "start_time", 0),
                    end_time=getattr(topic_segment, "end_time", duration),
                    topics=[t.topic for t in topic_segment.topics],
                    confidence_scores=[getattr(t, "confidence_score", 0.0) for t in topic_segment.topics],
                )
                result.chapters.append(chapter)

            for topic_segment in response.results.topics.segments:
                for topic in topic_segment.topics:
                    tname = topic.topic
                    result.topics[tname] = result.topics.get(tname, 0) + 1

        # Intents
        if hasattr(response.results, "intents") and response.results.intents:
            for segment in response.results.intents.segments:
                for intent in segment.intents:
                    result.intents.append(intent.intent)

        # Sentiments
        if hasattr(response.results, "sentiments") and response.results.sentiments:
            for segment in response.results.sentiments.segments:
                if hasattr(segment, "sentiment"):
                    s = segment.sentiment
                    result.sentiment_distribution[s] = result.sentiment_distribution.get(s, 0) + 1

        # Utterances + Speakers
        if hasattr(response.results, "utterances") and response.results.utterances:
            speaker_times: Dict[int, float] = {}
            for utterance in response.results.utterances:
                speaker_id = utterance.speaker
                duration_segment = utterance.end - utterance.start
                speaker_times[speaker_id] = speaker_times.get(speaker_id, 0.0) + duration_segment

                result.utterances.append(
                    TranscriptionUtterance(
                        speaker=speaker_id,
                        start=utterance.start,
                        end=utterance.end,
                        text=utterance.transcript,
                    )
                )

            # Speaker summaries
            safe_total = duration if duration and duration > 0 else 1.0
            for speaker_id, total_time in speaker_times.items():
                percentage = (total_time / safe_total) * 100 if safe_total > 0 else 0.0
                result.speakers.append(
                    TranscriptionSpeaker(
                        id=speaker_id, total_time=total_time, percentage=percentage
                    )
                )

        return result

    def _log_file_info(self, audio_file_path: Path) -> None:
        try:
            file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
        except Exception:
            # Non-fatal
            logger.debug("Could not determine file size for logging.")

    def __init__(
        self,
        api_key: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the transcriber with API key and configurations.

        Args:
            api_key: Optional Deepgram API key. If None, uses Config.DEEPGRAM_API_KEY
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration
        """
        # Use standardized provider initialization
        retry_config, circuit_config = ProviderInitializer.initialize_provider_configs(
            provider_name="Deepgram",
            retry_config=retry_config,
            circuit_config=circuit_config
            )

        if circuit_config is None:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                expected_exception_types=Config.CIRCUIT_BREAKER_EXPECTED_EXCEPTION_TYPES,
            )

        super().__init__(api_key, circuit_config, retry_config)
        self.api_key = api_key or Config.DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY not found. Set it as environment variable or pass to constructor."
            )

    def validate_configuration(self) -> bool:
        """Validate that Deepgram is properly configured.

        Returns:
            True if configuration is valid, False otherwise
        """
        return bool(self.api_key)

    def get_provider_name(self) -> str:
        """Get the name of this transcription provider.

        Returns:
            Human-readable name of the provider
        """
        return "Deepgram Nova 3"

    def get_supported_features(self) -> List[str]:
        """Get list of features supported by Deepgram Nova 3.

        Returns:
            List of feature names supported by this provider
        """
        return [
            "speaker_diarization",
            "topic_detection",
            "intent_analysis",
            "sentiment_analysis",
            "timestamps",
            "summarization",
            "language_detection",
            "punctuation",
            "paragraphs",
            "smart_format",
        ]

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    async def health_check_async(self) -> Dict[str, Any]:
        """Perform health check for Deepgram service.

        Returns:
            Dictionary containing health status information
        """
        start_time = time.time()

        try:
            # Import Deepgram SDK
            from deepgram import DeepgramClient

            # Validate API key format (Deepgram keys are typically 40+ characters)
            if not self.api_key or len(self.api_key) < 20:
                return {
                    "healthy": False,
                    "status": "invalid_api_key",
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "details": {
                        "provider": "Deepgram Nova 3",
                        "error": "API key appears to be invalid or missing",
                    },
                }

            # Try to create a client instance
            try:
                DeepgramClient(self.api_key)
                response_time = (time.time() - start_time) * 1000

                return {
                    "healthy": True,
                    "status": "operational",
                    "response_time_ms": response_time,
                    "details": {
                        "provider": "Deepgram Nova 3",
                        "api_accessible": True,
                        "authentication": "key_format_valid",
                        "note": "Health check validates SDK and key format only",
                    },
                }
            except Exception as client_error:
                return {
                    "healthy": False,
                    "status": "client_creation_failed",
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "details": {
                        "provider": "Deepgram Nova 3",
                        "error": f"Failed to create client: {client_error!s}",
                    },
                }

        except ImportError:
            return {
                "healthy": False,
                "status": "sdk_not_available",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {"provider": "Deepgram Nova 3", "error": "Deepgram SDK not installed"},
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "response_time_ms": (time.time() - start_time) * 1000,
                "details": {
                    "provider": "Deepgram Nova 3",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }

    async def _transcribe_impl(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Internal transcription implementation without retry/circuit breaker logic.

        Args:
            audio_file_path: Path to audio file
            language: Language code for transcription

        Returns:
            TranscriptionResult with all features, or None if failed
        """
        # Validate audio file
        validated_path = safe_validate_audio_file(audio_file_path, provider_name="deepgram")
        if validated_path is None:
            return None
        audio_file_path = validated_path

        try:
            logger.info(f"Starting Deepgram Nova 3 transcription: {audio_file_path}")
            self._log_file_info(audio_file_path)

            # Build request
            client = self._create_client()
            options = self._build_options(language)
            mimetype = self._detect_mimetype(audio_file_path)

            try:
                buffer = self._read_audio_file(audio_file_path)
            except (OSError, PermissionError) as e:
                logger.error(f"Failed to open audio file: {e}")
                return None

            # Submit job
            logger.info("Sending to Deepgram Nova 3...")
            response = self._submit_transcription_job(client, buffer, mimetype, options)
            logger.info("Transcription completed successfully")

            # Parse and return
            return self._parse_response(response, audio_file_path, language)

        except ImportError as e:
            logger.error(f"Deepgram SDK not installed: {e}")
            raise ConnectionError(f"Deepgram SDK not available: {e}") from e
        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {e}")
            raise ValueError(f"Audio file not found: {e}") from e
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {e}")
            raise ValueError(f"Permission denied: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid configuration or input: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Network connection error: {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Transcription request timed out: {e}")
            raise
        except OSError as e:
            logger.error(f"System error during transcription: {e}")
            raise ConnectionError(f"System error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            raise ConnectionError(f"Unexpected error: {e}") from e

    def transcribe(
        self, audio_file_path: Path, language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """Synchronous wrapper for transcription using asyncio.run.

        This avoids edge cases when external code patches event loop functions
        and ensures the coroutine is properly awaited.
        """
        try:
            return asyncio.run(self.transcribe_async(audio_file_path, language))
        except (RuntimeError, ValueError, ImportError, OSError) as e:
            # If already in a running loop (unlikely in CLI use), fall back to a new loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.transcribe_async(audio_file_path, language))
            except Exception as inner:
                logger.error(f"Synchronous transcription failed: {e or inner}")
                return None
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Unexpected error in synchronous transcription: {e}")
            return None

    def save_result_to_file(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save transcription result to formatted text file.

        Args:
            result: TranscriptionResult to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("DEEPGRAM NOVA 3 TRANSCRIPTION & ANALYSIS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio File: {os.path.basename(result.audio_file)}\n")
            duration_text = f"Audio Duration: {result.duration:.2f} seconds ({self._format_time(result.duration)})"
            f.write(f"{duration_text}\n")
            f.write("=" * 70 + "\n\n")

            # Executive summary
            if result.summary:
                f.write("EXECUTIVE SUMMARY:\n")
                f.write("-" * 50 + "\n")
                f.write(result.summary + "\n\n")
                f.write("=" * 70 + "\n\n")

            # Chapters
            if result.chapters:
                f.write("CONTENT CHAPTERS:\n")
                f.write("-" * 50 + "\n")
                for i, chapter in enumerate(result.chapters, 1):
                    start = self._format_time(chapter.start_time)
                    end = self._format_time(chapter.end_time)
                    f.write(f"\nChapter {i}: [{start} - {end}]\n")
                    for j, topic in enumerate(chapter.topics):
                        confidence = (
                            chapter.confidence_scores[j]
                            if j < len(chapter.confidence_scores)
                            else 0
                        )
                        f.write(f"  • {topic} ({confidence:.1%})\n")
                f.write("\n" + "=" * 70 + "\n\n")

            # Key topics
            if result.topics:
                f.write("KEY TOPICS DISCUSSED:\n")
                f.write("-" * 50 + "\n")
                for topic, count in sorted(result.topics.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"• {topic} (mentioned {count} times)\n")
                f.write("\n" + "=" * 70 + "\n\n")

            # Intents
            if result.intents:
                f.write("DETECTED INTENTS:\n")
                f.write("-" * 50 + "\n")
                for intent in set(result.intents):  # Remove duplicates
                    f.write(f"• {intent}\n")
                f.write("\n" + "=" * 70 + "\n\n")

            # Sentiment
            if result.sentiment_distribution:
                f.write("SENTIMENT ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                for sentiment, count in result.sentiment_distribution.items():
                    f.write(f"• {sentiment.capitalize()}: {count} segments\n")
                f.write("\n" + "=" * 70 + "\n\n")

            # Speaker-separated transcript
            if result.utterances:
                f.write("SPEAKER-SEPARATED TRANSCRIPT:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Speakers Detected: {len(result.speakers)}\n\n")

                current_speaker = None
                for utterance in result.utterances:
                    if current_speaker != utterance.speaker:
                        f.write(
                            f"\n[{self._format_time(utterance.start)}] Speaker {utterance.speaker + 1}:\n"
                        )
                        current_speaker = utterance.speaker
                    f.write(f"{utterance.text}\n")

                # Speaker time distribution
                f.write("\n" + "-" * 50 + "\n")
                f.write("SPEAKER TIME DISTRIBUTION:\n")
                for speaker in result.speakers:
                    time_str = self._format_time(speaker.total_time)
                    pct = speaker.percentage
                    f.write(f"• Speaker {speaker.id + 1}: {time_str} ({pct:.1f}%)\n")

                f.write("\n" + "=" * 70 + "\n\n")

            # Full transcript
            f.write("FULL TRANSCRIPT:\n")
            f.write("-" * 50 + "\n\n")
            f.write(result.transcript)

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF TRANSCRIPTION\n")

        logger.info(f"Transcription saved to: {output_path}")
