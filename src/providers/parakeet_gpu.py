"""GPU management utilities for Parakeet transcription provider."""
from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# GPU availability check
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU features will be disabled.")


class ParakeetGPUError(Exception):
    """Raised when GPU operations fail."""

    pass


class GPUManager:
    """Manages GPU resources for Parakeet models."""

    def __init__(self):
        """Initialize GPU manager with device detection."""
        self._device = None
        self._device_id = None
        if TORCH_AVAILABLE:
            self._device = self._detect_best_device()
            if self._device.startswith("cuda"):
                self._device_id = int(self._device.split(":")[-1]) if ":" in self._device else 0

    @property
    def device(self) -> str:
        """Get the current device string (e.g., 'cuda:0', 'cpu')."""
        if not self._device:
            return "cpu"
        return self._device

    @property
    def device_id(self) -> Optional[int]:
        """Get the CUDA device ID if using GPU."""
        return self._device_id

    def _detect_best_device(self) -> str:
        """Detect the best available device for model execution.

        Returns:
            Device string (e.g., 'cuda:0', 'mps', 'cpu')
        """
        if not TORCH_AVAILABLE:
            return "cpu"

        try:
            if torch.cuda.is_available():
                # Find GPU with most free memory
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    return "cpu"

                best_device = 0
                max_free_memory = 0

                for i in range(device_count):
                    free_memory = torch.cuda.mem_get_info(i)[0]
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = i

                logger.info(
                    f"Selected CUDA device {best_device} with {max_free_memory / 1e9:.2f}GB free memory"
                )
                return f"cuda:{best_device}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using Apple Metal Performance Shaders (MPS)")
                return "mps"
        except Exception as e:
            logger.warning(f"Error detecting GPU device: {e}")

        return "cpu"

    def get_available_memory(self) -> Optional[int]:
        """Get available memory on current device in bytes.

        Returns:
            Available memory in bytes, or None if cannot be determined
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            if self._device and self._device.startswith("cuda"):
                if self._device_id is not None:
                    free, total = torch.cuda.mem_get_info(self._device_id)
                    return free
            elif self._device == "mps":
                # MPS doesn't provide direct memory query
                # Return a conservative estimate
                return 4 * 1024 * 1024 * 1024  # 4GB
        except Exception as e:
            logger.warning(f"Could not get available memory: {e}")

        return None

    def can_allocate_model(self, estimated_model_size: int) -> bool:
        """Check if there's enough memory to allocate a model.

        Args:
            estimated_model_size: Estimated model size in bytes

        Returns:
            True if model can likely be allocated
        """
        available = self.get_available_memory()
        if available is None:
            # If we can't determine memory, optimistically return True
            # The actual allocation will fail if insufficient
            return True

        # Leave 500MB buffer for operations
        buffer = 500 * 1024 * 1024
        return available > (estimated_model_size + buffer)

    def cleanup_gpu_memory(self) -> None:
        """Force GPU memory cleanup."""
        if not TORCH_AVAILABLE:
            return

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared")
        except Exception as e:
            logger.warning(f"Error cleaning up GPU memory: {e}")
