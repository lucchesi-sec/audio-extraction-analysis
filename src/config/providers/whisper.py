"""Whisper provider configuration."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BaseConfig, ConfigurationSchema
from ..performance import get_performance_config

logger = logging.getLogger(__name__)


class WhisperModel(Enum):
    """Available Whisper models."""

    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE = "large"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class WhisperDevice(Enum):
    """Device options for Whisper."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class WhisperComputeType(Enum):
    """Compute type options for faster-whisper."""

    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT16 = "int16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    AUTO = "auto"


class WhisperConfig(BaseConfig):
    """Whisper-specific configuration."""

    def __init__(self):
        """Initialize Whisper configuration."""
        super().__init__()

        # Get dependencies
        self._performance = get_performance_config()

        # Model settings
        model_str = self.get_value("WHISPER_MODEL", "base")
        self.model = self._parse_model(model_str)
        self.model_dir = Path(self.get_value("WHISPER_MODEL_DIR", "~/.cache/whisper")).expanduser()
        self.download_root = Path(self.get_value("WHISPER_DOWNLOAD_ROOT", str(self.model_dir)))

        # Device settings
        device_str = self.get_value("WHISPER_DEVICE", "auto")
        self.device = self._parse_device(device_str)
        self.device_index = self._parse_device_index(self.get_value("WHISPER_DEVICE_INDEX", "0"))

        # Compute settings (for faster-whisper)
        compute_str = self.get_value("WHISPER_COMPUTE_TYPE", "auto")
        self.compute_type = self._parse_compute_type(compute_str)
        self.num_workers = int(self.get_value("WHISPER_NUM_WORKERS", "1"))
        self.cpu_threads = int(self.get_value("WHISPER_CPU_THREADS", "0"))  # 0 = auto

        # Language settings
        self.language = self.get_value("WHISPER_LANGUAGE")  # None = auto-detect
        self.task = self.get_value("WHISPER_TASK", "transcribe")  # transcribe or translate

        # Decoding settings
        self.temperature = self._parse_temperature(self.get_value("WHISPER_TEMPERATURE", "0"))
        self.best_of = int(self.get_value("WHISPER_BEST_OF", "5"))
        self.beam_size = int(self.get_value("WHISPER_BEAM_SIZE", "5"))
        self.patience = float(self.get_value("WHISPER_PATIENCE", "1.0"))
        self.length_penalty = float(self.get_value("WHISPER_LENGTH_PENALTY", "1.0"))
        self.suppress_tokens = self._parse_suppress_tokens(
            self.get_value("WHISPER_SUPPRESS_TOKENS", "-1")
        )
        self.suppress_blank = self.parse_bool(self.get_value("WHISPER_SUPPRESS_BLANK", "true"))

        # Word timestamps
        self.word_timestamps = self.parse_bool(self.get_value("WHISPER_WORD_TIMESTAMPS", "false"))
        self.prepend_punctuations = self.get_value("WHISPER_PREPEND_PUNCTUATIONS", '"\'"¿([{-')
        self.append_punctuations = self.get_value(
            "WHISPER_APPEND_PUNCTUATIONS", '"\'.。,，!！?？:：")]}、'
        )

        # VAD (Voice Activity Detection)
        self.vad_filter = self.parse_bool(self.get_value("WHISPER_VAD_FILTER", "false"))
        self.vad_parameters = {
            "threshold": float(self.get_value("WHISPER_VAD_THRESHOLD", "0.6")),
            "min_speech_duration_ms": int(self.get_value("WHISPER_VAD_MIN_SPEECH_MS", "250")),
            "max_speech_duration_s": float(self.get_value("WHISPER_VAD_MAX_SPEECH_S", "0")),
            "min_silence_duration_ms": int(self.get_value("WHISPER_VAD_MIN_SILENCE_MS", "2000")),
            "window_size_samples": int(self.get_value("WHISPER_VAD_WINDOW_SIZE", "1024")),
            "speech_pad_ms": int(self.get_value("WHISPER_VAD_SPEECH_PAD_MS", "400")),
        }

        # Chunking settings
        self.chunk_length = int(self.get_value("WHISPER_CHUNK_LENGTH", "30"))  # seconds
        self.chunk_overlap = int(self.get_value("WHISPER_CHUNK_OVERLAP", "5"))  # seconds

        # Output settings
        self.fp16 = self.parse_bool(self.get_value("WHISPER_FP16", "true"))
        self.condition_on_previous_text = self.parse_bool(
            self.get_value("WHISPER_CONDITION_ON_PREVIOUS", "true")
        )
        self.compression_ratio_threshold = float(
            self.get_value("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
        )
        self.logprob_threshold = float(self.get_value("WHISPER_LOGPROB_THRESHOLD", "-1.0"))
        self.no_speech_threshold = float(self.get_value("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))

        # Initial prompt
        self.initial_prompt = self.get_value("WHISPER_INITIAL_PROMPT")

        # Hallucination silence threshold
        self.hallucination_silence_threshold = float(
            self.get_value("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "0.0")
        )

        # Performance settings
        self.timeout = int(self.get_value("WHISPER_TIMEOUT", "3600"))  # 1 hour default
        self.max_file_size = int(self.get_value("WHISPER_MAX_FILE_SIZE", "1073741824"))  # 1GB

        # Memory settings
        self.max_memory_gb = float(self.get_value("WHISPER_MAX_MEMORY_GB", "0"))  # 0 = unlimited
        self.offload_to_cpu = self.parse_bool(self.get_value("WHISPER_OFFLOAD_TO_CPU", "false"))

        # Faster-whisper specific
        self.use_faster_whisper = self.parse_bool(self.get_value("WHISPER_USE_FASTER", "false"))
        self.faster_whisper_threads = int(self.get_value("WHISPER_FASTER_THREADS", "4"))

        # Cache settings
        self.enable_model_cache = self.parse_bool(
            self.get_value("WHISPER_ENABLE_MODEL_CACHE", "true")
        )
        self.cache_dir = Path(self.get_value("WHISPER_CACHE_DIR", "~/.cache/whisper")).expanduser()

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for directory in [self.model_dir, self.cache_dir, self.download_root]:
            directory.mkdir(parents=True, exist_ok=True)

    def _parse_model(self, model_str: str) -> WhisperModel:
        """Parse model string to enum.

        Args:
            model_str: Model string

        Returns:
            WhisperModel enum value
        """
        try:
            return WhisperModel(model_str.lower())
        except ValueError:
            logger.warning(f"Unknown Whisper model: {model_str}, using 'base'")
            return WhisperModel.BASE

    def _parse_device(self, device_str: str) -> WhisperDevice:
        """Parse device string to enum.

        Args:
            device_str: Device string

        Returns:
            WhisperDevice enum value
        """
        if device_str.lower() == "auto":
            # Auto-detect best available device
            try:
                import torch

                if torch.cuda.is_available():
                    return WhisperDevice.CUDA
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return WhisperDevice.MPS
                else:
                    return WhisperDevice.CPU
            except ImportError:
                return WhisperDevice.CPU

        try:
            return WhisperDevice(device_str.lower())
        except ValueError:
            logger.warning(f"Unknown device: {device_str}, using CPU")
            return WhisperDevice.CPU

    def _parse_device_index(self, index_str: str) -> Optional[int]:
        """Parse device index string.

        Args:
            index_str: Device index string

        Returns:
            Device index or None
        """
        try:
            index = int(index_str)
            return index if index >= 0 else None
        except (ValueError, TypeError):
            return None

    def _parse_compute_type(self, compute_str: str) -> WhisperComputeType:
        """Parse compute type string to enum.

        Args:
            compute_str: Compute type string

        Returns:
            WhisperComputeType enum value
        """
        if compute_str.lower() == "auto":
            # Auto-select based on device
            if self.device == WhisperDevice.CUDA:
                return WhisperComputeType.FLOAT16
            else:
                return WhisperComputeType.FLOAT32

        try:
            return WhisperComputeType(compute_str.lower())
        except ValueError:
            logger.warning(f"Unknown compute type: {compute_str}, using auto")
            return WhisperComputeType.AUTO

    def _parse_temperature(self, temp_str: str) -> List[float]:
        """Parse temperature string to list of floats.

        Args:
            temp_str: Temperature string (comma-separated or single value)

        Returns:
            List of temperature values
        """
        if not temp_str:
            return [0.0]

        if "," in temp_str:
            return [float(t.strip()) for t in temp_str.split(",")]
        else:
            return [float(temp_str)]

    def _parse_suppress_tokens(self, tokens_str: str) -> Optional[List[int]]:
        """Parse suppress tokens string.

        Args:
            tokens_str: Comma-separated token IDs or "-1" for default

        Returns:
            List of token IDs or None
        """
        if tokens_str == "-1":
            return None

        if tokens_str:
            return [int(t.strip()) for t in tokens_str.split(",") if t.strip()]

        return None

    def get_transcribe_options(self) -> Dict[str, Any]:
        """Get transcription options for Whisper.

        Returns:
            Dictionary of transcribe options
        """
        options = {
            "task": self.task,
            "temperature": self.temperature,
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "suppress_blank": self.suppress_blank,
            "condition_on_previous_text": self.condition_on_previous_text,
            "fp16": self.fp16 and self.device != WhisperDevice.CPU,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
        }

        if self.language:
            options["language"] = self.language

        if self.suppress_tokens:
            options["suppress_tokens"] = self.suppress_tokens

        if self.initial_prompt:
            options["initial_prompt"] = self.initial_prompt

        if self.word_timestamps:
            options["word_timestamps"] = True
            options["prepend_punctuations"] = self.prepend_punctuations
            options["append_punctuations"] = self.append_punctuations

        if self.hallucination_silence_threshold:
            options["hallucination_silence_threshold"] = self.hallucination_silence_threshold

        return options

    def get_vad_options(self) -> Dict[str, Any]:
        """Get VAD options if enabled.

        Returns:
            Dictionary of VAD options or empty dict
        """
        if not self.vad_filter:
            return {}

        return {"vad_filter": True, "vad_parameters": self.vad_parameters}

    def get_model_path(self) -> Path:
        """Get full path to model file.

        Returns:
            Path to model file
        """
        return self.model_dir / f"{self.model.value}.pt"

    def get_device_string(self) -> str:
        """Get device string for PyTorch.

        Returns:
            Device string
        """
        if self.device == WhisperDevice.AUTO:
            return self._parse_device("auto").value

        device_str = self.device.value
        if self.device_index is not None and self.device == WhisperDevice.CUDA:
            device_str = f"{device_str}:{self.device_index}"

        return device_str

    def estimate_memory_usage(self) -> float:
        """Estimate memory usage based on model size.

        Returns:
            Estimated memory usage in GB
        """
        # Approximate memory usage for models
        memory_map = {
            WhisperModel.TINY: 0.1,
            WhisperModel.TINY_EN: 0.1,
            WhisperModel.BASE: 0.2,
            WhisperModel.BASE_EN: 0.2,
            WhisperModel.SMALL: 0.5,
            WhisperModel.SMALL_EN: 0.5,
            WhisperModel.MEDIUM: 1.5,
            WhisperModel.MEDIUM_EN: 1.5,
            WhisperModel.LARGE: 3.0,
            WhisperModel.LARGE_V1: 3.0,
            WhisperModel.LARGE_V2: 3.0,
            WhisperModel.LARGE_V3: 3.0,
        }

        base_memory = memory_map.get(self.model, 1.0)

        # Add overhead for batch processing
        if self.num_workers > 1:
            base_memory *= 1.5

        return base_memory

    def validate_resources(self) -> bool:
        """Validate that system has sufficient resources.

        Returns:
            True if resources are sufficient
        """
        estimated_memory = self.estimate_memory_usage()

        if self.max_memory_gb > 0 and estimated_memory > self.max_memory_gb:
            logger.warning(
                f"Model {self.model.value} requires ~{estimated_memory:.1f}GB, "
                f"but limit is {self.max_memory_gb}GB"
            )
            return False

        return True

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for Whisper config
        """
        return ConfigurationSchema(
            name="WhisperConfig",
            required_fields=set(),
            optional_fields={
                "model": "base",
                "device": "auto",
                "compute_type": "auto",
                "task": "transcribe",
                "beam_size": 5,
                "fp16": True,
                "timeout": 3600,
            },
            validators={
                "model": lambda x: x in {m.value for m in WhisperModel},
                "device": lambda x: x in {d.value for d in WhisperDevice},
                "task": lambda x: x in {"transcribe", "translate"},
                "beam_size": lambda x: isinstance(x, int) and 1 <= x <= 20,
                "best_of": lambda x: isinstance(x, int) and x >= 1,
                "chunk_length": lambda x: isinstance(x, int) and x > 0,
                "num_workers": lambda x: isinstance(x, int) and x >= 1,
                "no_speech_threshold": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "model": self.model.value,
            "model_dir": str(self.model_dir),
            "device": self.device.value,
            "device_index": self.device_index,
            "compute_type": self.compute_type.value,
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "fp16": self.fp16,
            "chunk_length": self.chunk_length,
            "timeout": self.timeout,
            "max_file_size": self.max_file_size,
            "use_faster_whisper": self.use_faster_whisper,
            "estimated_memory_gb": self.estimate_memory_usage(),
        }


# Singleton instance getter
def get_whisper_config() -> WhisperConfig:
    """Get Whisper configuration instance.

    Returns:
        WhisperConfig singleton instance
    """
    return WhisperConfig.get_instance()
