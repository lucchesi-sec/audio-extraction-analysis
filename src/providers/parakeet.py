"""NVIDIA Parakeet STT transcription service.

This module re-exports the split Parakeet implementation for backward compatibility.
The implementation has been refactored into:
- parakeet_core.py: Main transcription logic
- parakeet_gpu.py: GPU management
- parakeet_cache.py: Model caching
- parakeet_audio.py: Audio preprocessing
"""
from __future__ import annotations

from .parakeet_audio import AudioPreprocessor, ParakeetAudioError
from .parakeet_cache import ParakeetModelCache, ParakeetModelError

# Re-export everything from the split modules for backward compatibility
from .parakeet_core import PARAKEET_MODELS, ParakeetError, ParakeetMetrics, ParakeetTranscriber
from .parakeet_gpu import GPUManager, ParakeetGPUError

# Maintain backward compatibility by exposing all classes at module level
__all__ = [
    "PARAKEET_MODELS",
    "AudioPreprocessor",
    "GPUManager",
    "ParakeetAudioError",
    "ParakeetError",
    "ParakeetGPUError",
    "ParakeetMetrics",
    "ParakeetModelCache",
    "ParakeetModelError",
    "ParakeetTranscriber",
]
