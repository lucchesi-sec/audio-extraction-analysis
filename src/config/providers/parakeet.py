"""Parakeet provider configuration."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseConfig, ConfigurationSchema
from ..performance import get_performance_config

logger = logging.getLogger(__name__)


class ParakeetModel(Enum):
    """Available Parakeet models."""

    STT_EN_CONFORMER_CTC_LARGE = "stt_en_conformer_ctc_large"
    STT_EN_CONFORMER_CTC_MEDIUM = "stt_en_conformer_ctc_medium"
    STT_EN_CONFORMER_CTC_SMALL = "stt_en_conformer_ctc_small"
    STT_EN_CONFORMER_TRANSDUCER_LARGE = "stt_en_conformer_transducer_large"
    STT_EN_CONFORMER_TRANSDUCER_MEDIUM = "stt_en_conformer_transducer_medium"
    STT_EN_CONFORMER_TRANSDUCER_SMALL = "stt_en_conformer_transducer_small"
    STT_EN_FASTCONFORMER_CTC_LARGE = "stt_en_fastconformer_ctc_large"
    STT_EN_FASTCONFORMER_TRANSDUCER_LARGE = "stt_en_fastconformer_transducer_large"
    STT_EN_QUARTZNET15X5 = "stt_en_quartznet15x5"
    STT_EN_JASPER10X5DR = "stt_en_jasper10x5dr"
    STT_EN_CITRINET_256 = "stt_en_citrinet_256"
    STT_EN_CITRINET_512 = "stt_en_citrinet_512"
    STT_EN_CITRINET_1024 = "stt_en_citrinet_1024"


class ParakeetDecoder(Enum):
    """Decoder types for Parakeet."""

    GREEDY = "greedy"
    BEAM = "beam"
    FLASHLIGHT = "flashlight"
    PYCTCDECODE = "pyctcdecode"


class ParakeetConfig(BaseConfig):
    """Parakeet (NVIDIA NeMo) specific configuration."""

    def __init__(self):
        """Initialize Parakeet configuration."""
        super().__init__()

        # Get dependencies
        self._performance = get_performance_config()

        # Model settings
        model_str = self.get_value("PARAKEET_MODEL", "stt_en_conformer_ctc_large")
        self.model = self._parse_model(model_str)
        self.model_cache_dir = Path(
            self.get_value("PARAKEET_MODEL_CACHE_DIR", "~/.cache/parakeet")
        ).expanduser()
        self.pretrained = self.parse_bool(self.get_value("PARAKEET_PRETRAINED", "true"))

        # Device settings
        self.device = self.get_value("PARAKEET_DEVICE", "auto")
        self.device_id = self._parse_device_id(self.get_value("PARAKEET_DEVICE_ID", "0"))

        # Decoder settings
        decoder_str = self.get_value("PARAKEET_DECODER", "greedy")
        self.decoder = self._parse_decoder(decoder_str)
        self.beam_size = int(self.get_value("PARAKEET_BEAM_SIZE", "10"))
        self.beam_alpha = float(self.get_value("PARAKEET_BEAM_ALPHA", "2.0"))
        self.beam_beta = float(self.get_value("PARAKEET_BEAM_BETA", "0.0"))
        self.lm_path = self.get_value("PARAKEET_LM_PATH")  # Language model path

        # Processing settings
        self.batch_size = int(self.get_value("PARAKEET_BATCH_SIZE", "8"))
        self.chunk_length = int(self.get_value("PARAKEET_CHUNK_LENGTH", "30"))  # seconds
        self.overlap_length = int(self.get_value("PARAKEET_OVERLAP_LENGTH", "2"))  # seconds
        self.use_fp16 = self.parse_bool(self.get_value("PARAKEET_USE_FP16", "true"))
        self.normalize_audio = self.parse_bool(self.get_value("PARAKEET_NORMALIZE_AUDIO", "true"))

        # Audio preprocessing
        self.sample_rate = int(self.get_value("PARAKEET_SAMPLE_RATE", "16000"))
        self.window_size = float(self.get_value("PARAKEET_WINDOW_SIZE", "0.02"))
        self.window_stride = float(self.get_value("PARAKEET_WINDOW_STRIDE", "0.01"))
        self.window_type = self.get_value("PARAKEET_WINDOW_TYPE", "hann")
        self.n_fft = int(self.get_value("PARAKEET_N_FFT", "512"))
        self.n_mels = int(self.get_value("PARAKEET_N_MELS", "64"))
        self.freq_min = float(self.get_value("PARAKEET_FREQ_MIN", "0.0"))
        self.freq_max = float(self.get_value("PARAKEET_FREQ_MAX", "8000.0"))

        # Feature extraction
        self.features = self.get_value("PARAKEET_FEATURES", "mfcc")  # mfcc, fbank, spectrogram
        self.normalize_features = self.parse_bool(
            self.get_value("PARAKEET_NORMALIZE_FEATURES", "true")
        )
        self.dither = float(self.get_value("PARAKEET_DITHER", "1e-5"))
        self.preemphasis = float(self.get_value("PARAKEET_PREEMPHASIS", "0.97"))

        # Punctuation and formatting
        self.add_punctuation = self.parse_bool(self.get_value("PARAKEET_ADD_PUNCTUATION", "true"))
        self.preserve_word_case = self.parse_bool(
            self.get_value("PARAKEET_PRESERVE_WORD_CASE", "false")
        )
        self.replace_words = self._parse_replace_words(self.get_value("PARAKEET_REPLACE_WORDS", ""))

        # Confidence settings
        self.return_confidence = self.parse_bool(
            self.get_value("PARAKEET_RETURN_CONFIDENCE", "true")
        )
        self.confidence_threshold = float(self.get_value("PARAKEET_CONFIDENCE_THRESHOLD", "0.5"))
        self.word_confidence = self.parse_bool(self.get_value("PARAKEET_WORD_CONFIDENCE", "false"))

        # VAD settings
        self.use_vad = self.parse_bool(self.get_value("PARAKEET_USE_VAD", "false"))
        self.vad_threshold = float(self.get_value("PARAKEET_VAD_THRESHOLD", "0.5"))
        self.vad_window_length = float(self.get_value("PARAKEET_VAD_WINDOW_LENGTH", "0.15"))
        self.vad_shift_length = float(self.get_value("PARAKEET_VAD_SHIFT_LENGTH", "0.01"))

        # Speaker diarization
        self.diarize = self.parse_bool(self.get_value("PARAKEET_DIARIZE", "false"))
        self.max_speakers = int(self.get_value("PARAKEET_MAX_SPEAKERS", "10"))
        self.oracle_num_speakers = self._parse_oracle_speakers(
            self.get_value("PARAKEET_ORACLE_NUM_SPEAKERS")
        )

        # Performance settings
        self.timeout = int(self.get_value("PARAKEET_TIMEOUT", "3600"))
        self.max_file_size = int(self.get_value("PARAKEET_MAX_FILE_SIZE", "1073741824"))  # 1GB
        self.num_workers = int(self.get_value("PARAKEET_NUM_WORKERS", "4"))
        self.pin_memory = self.parse_bool(self.get_value("PARAKEET_PIN_MEMORY", "true"))

        # Cache settings
        self.enable_cache = self.parse_bool(self.get_value("PARAKEET_ENABLE_CACHE", "true"))
        self.cache_dir = Path(
            self.get_value("PARAKEET_CACHE_DIR", "~/.cache/parakeet")
        ).expanduser()

        # Optimization
        self.torch_compile = self.parse_bool(self.get_value("PARAKEET_TORCH_COMPILE", "false"))
        self.torch_compile_mode = self.get_value("PARAKEET_TORCH_COMPILE_MODE", "default")
        self.enable_onnx = self.parse_bool(self.get_value("PARAKEET_ENABLE_ONNX", "false"))
        self.onnx_precision = self.get_value("PARAKEET_ONNX_PRECISION", "fp16")

        # Metrics
        self.compute_wer = self.parse_bool(self.get_value("PARAKEET_COMPUTE_WER", "false"))
        self.compute_cer = self.parse_bool(self.get_value("PARAKEET_COMPUTE_CER", "false"))

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for directory in [self.model_cache_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _parse_model(self, model_str: str) -> ParakeetModel:
        """Parse model string to enum.

        Args:
            model_str: Model string

        Returns:
            ParakeetModel enum value
        """
        try:
            return ParakeetModel(model_str.lower())
        except ValueError:
            logger.warning(f"Unknown Parakeet model: {model_str}, using stt_en_conformer_ctc_large")
            return ParakeetModel.STT_EN_CONFORMER_CTC_LARGE

    def _parse_decoder(self, decoder_str: str) -> ParakeetDecoder:
        """Parse decoder string to enum.

        Args:
            decoder_str: Decoder string

        Returns:
            ParakeetDecoder enum value
        """
        try:
            return ParakeetDecoder(decoder_str.lower())
        except ValueError:
            logger.warning(f"Unknown decoder: {decoder_str}, using greedy")
            return ParakeetDecoder.GREEDY

    def _parse_device_id(self, device_str: str) -> Optional[int]:
        """Parse device ID string.

        Args:
            device_str: Device ID string

        Returns:
            Device ID or None
        """
        try:
            device_id = int(device_str)
            return device_id if device_id >= 0 else None
        except (ValueError, TypeError):
            return None

    def _parse_replace_words(self, replace_str: str) -> Dict[str, str]:
        """Parse word replacement string.

        Args:
            replace_str: Comma-separated pairs like "word1:replacement1,word2:replacement2"

        Returns:
            Dictionary of replacements
        """
        if not replace_str:
            return {}

        replacements = {}
        for pair in replace_str.split(","):
            if ":" in pair:
                old, new = pair.split(":", 1)
                replacements[old.strip()] = new.strip()

        return replacements

    def _parse_oracle_speakers(self, speakers_str: Optional[str]) -> Optional[int]:
        """Parse oracle number of speakers.

        Args:
            speakers_str: Number of speakers string

        Returns:
            Number of speakers or None
        """
        if not speakers_str:
            return None

        try:
            return int(speakers_str)
        except ValueError:
            return None

    def get_device_string(self) -> str:
        """Get device string for PyTorch.

        Returns:
            Device string
        """
        if self.device.lower() == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = self.device

        if device == "cuda" and self.device_id is not None:
            return f"{device}:{self.device_id}"

        return device

    def get_decoder_config(self) -> Dict[str, Any]:
        """Get decoder configuration.

        Returns:
            Decoder configuration dictionary
        """
        config = {"decoder": self.decoder.value}

        if self.decoder in [
            ParakeetDecoder.BEAM,
            ParakeetDecoder.FLASHLIGHT,
            ParakeetDecoder.PYCTCDECODE,
        ]:
            config.update(
                {
                    "beam_size": self.beam_size,
                    "beam_alpha": self.beam_alpha,
                    "beam_beta": self.beam_beta,
                }
            )

            if self.lm_path:
                config["lm_path"] = self.lm_path

        return config

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get audio preprocessing configuration.

        Returns:
            Preprocessing configuration dictionary
        """
        return {
            "sample_rate": self.sample_rate,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "window": self.window_type,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "features": self.features,
            "normalize": self.normalize_features,
            "dither": self.dither,
            "preemphasis": self.preemphasis,
        }

    def get_vad_config(self) -> Dict[str, Any]:
        """Get VAD configuration if enabled.

        Returns:
            VAD configuration dictionary or empty dict
        """
        if not self.use_vad:
            return {}

        return {
            "vad_threshold": self.vad_threshold,
            "vad_window_length": self.vad_window_length,
            "vad_shift_length": self.vad_shift_length,
        }

    def get_diarization_config(self) -> Dict[str, Any]:
        """Get speaker diarization configuration if enabled.

        Returns:
            Diarization configuration dictionary or empty dict
        """
        if not self.diarize:
            return {}

        config = {"max_speakers": self.max_speakers}

        if self.oracle_num_speakers:
            config["oracle_num_speakers"] = self.oracle_num_speakers

        return config

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for NeMo.

        Returns:
            Model configuration dictionary
        """
        return {
            "model_name": self.model.value,
            "pretrained": self.pretrained,
            "cache_dir": str(self.model_cache_dir),
            "device": self.get_device_string(),
            "batch_size": self.batch_size,
            "use_fp16": self.use_fp16 and "cuda" in self.get_device_string(),
            **self.get_decoder_config(),
            **self.get_preprocessing_config(),
        }

    def estimate_memory_usage(self) -> float:
        """Estimate memory usage based on model.

        Returns:
            Estimated memory usage in GB
        """
        # Approximate memory usage for models
        memory_map = {
            "small": 0.5,
            "medium": 1.0,
            "large": 2.0,
            "quartznet": 0.3,
            "jasper": 0.4,
            "citrinet_256": 0.6,
            "citrinet_512": 1.2,
            "citrinet_1024": 2.5,
        }

        # Determine model size category
        model_name = self.model.value.lower()
        if "small" in model_name:
            base_memory = memory_map["small"]
        elif "medium" in model_name:
            base_memory = memory_map["medium"]
        elif "large" in model_name:
            base_memory = memory_map["large"]
        elif "quartznet" in model_name:
            base_memory = memory_map["quartznet"]
        elif "jasper" in model_name:
            base_memory = memory_map["jasper"]
        elif "citrinet_256" in model_name:
            base_memory = memory_map["citrinet_256"]
        elif "citrinet_512" in model_name:
            base_memory = memory_map["citrinet_512"]
        elif "citrinet_1024" in model_name:
            base_memory = memory_map["citrinet_1024"]
        else:
            base_memory = 1.0  # Default

        # Adjust for batch size
        memory_multiplier = 1.0 + (self.batch_size - 1) * 0.2

        return base_memory * memory_multiplier

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for Parakeet config
        """
        return ConfigurationSchema(
            name="ParakeetConfig",
            required_fields=set(),
            optional_fields={
                "model": "stt_en_conformer_ctc_large",
                "device": "auto",
                "decoder": "greedy",
                "batch_size": 8,
                "chunk_length": 30,
                "use_fp16": True,
                "sample_rate": 16000,
                "timeout": 3600,
            },
            validators={
                "model": lambda x: x in {m.value for m in ParakeetModel},
                "decoder": lambda x: x in {d.value for d in ParakeetDecoder},
                "batch_size": lambda x: isinstance(x, int) and 1 <= x <= 128,
                "beam_size": lambda x: isinstance(x, int) and 1 <= x <= 100,
                "chunk_length": lambda x: isinstance(x, int) and x > 0,
                "sample_rate": lambda x: isinstance(x, int)
                and x in {8000, 16000, 22050, 44100, 48000},
                "confidence_threshold": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
                "vad_threshold": lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "model": self.model.value,
            "model_cache_dir": str(self.model_cache_dir),
            "device": self.device,
            "device_id": self.device_id,
            "decoder": self.decoder.value,
            "beam_size": self.beam_size,
            "batch_size": self.batch_size,
            "chunk_length": self.chunk_length,
            "use_fp16": self.use_fp16,
            "sample_rate": self.sample_rate,
            "add_punctuation": self.add_punctuation,
            "return_confidence": self.return_confidence,
            "use_vad": self.use_vad,
            "diarize": self.diarize,
            "timeout": self.timeout,
            "max_file_size": self.max_file_size,
            "estimated_memory_gb": self.estimate_memory_usage(),
        }


# Singleton instance getter
def get_parakeet_config() -> ParakeetConfig:
    """Get Parakeet configuration instance.

    Returns:
        ParakeetConfig singleton instance
    """
    return ParakeetConfig.get_instance()
