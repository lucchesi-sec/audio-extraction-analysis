"""Model caching utilities for Parakeet transcription provider."""
from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

from .parakeet_gpu import TORCH_AVAILABLE, GPUManager

logger = logging.getLogger(__name__)

# Lazy import for NeMo to avoid import-time failures in environments without it
nemo_asr = None
NEMO_AVAILABLE = None

def _ensure_nemo() -> bool:
    global NEMO_AVAILABLE, nemo_asr
    if NEMO_AVAILABLE is not None:
        return NEMO_AVAILABLE
    try:
        import nemo.collections.asr as _nemo_asr  # type: ignore
        nemo_asr = _nemo_asr
        NEMO_AVAILABLE = True
    except Exception:
        NEMO_AVAILABLE = False
        logger.warning("NeMo toolkit not available. Parakeet features will be disabled.")
    return NEMO_AVAILABLE


class ParakeetModelError(Exception):
    """Raised when model loading fails."""

    pass


class ParakeetModelCache:
    """Singleton cache for Parakeet ASR models.

    Implements LRU caching with GPU memory management.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        """Initialize the cache (called once)."""
        if self._initialized:
            return

        self._models = {}  # model_name -> (model, last_used_time)
        self._model_sizes = {}  # model_name -> size_in_bytes
        self._max_cache_size = 3  # Maximum number of models to cache
        self._cache_lock = Lock()
        self._loading_locks = {}  # Per-model loading locks
        self._gpu_manager = GPUManager()
        self._initialized = True

        logger.info(f"ParakeetModelCache initialized with device: {self._gpu_manager.device}")

    def __init__(self):
        """Initialize cache if not already done."""
        if not hasattr(self, "_initialized") or not self._initialized:
            self._initialize()

    def get_model(self, model_name: str, force_reload: bool = False) -> Optional[Any]:
        """Get a model from cache or load it.

        Args:
            model_name: Name of the Parakeet model
            force_reload: Force reload even if cached

        Returns:
            Loaded model or None if loading fails
        """
        if not _ensure_nemo():
            logger.error("NeMo toolkit not available")
            return None

        # Get or create a loading lock for this specific model
        with self._cache_lock:
            if model_name not in self._loading_locks:
                self._loading_locks[model_name] = Lock()
            loading_lock = self._loading_locks[model_name]

        # Use the model-specific lock for loading
        with loading_lock:
            # Check cache first (inside the model lock)
            with self._cache_lock:
                if not force_reload and model_name in self._models:
                    model, _ = self._models[model_name]
                    self._models[model_name] = (model, time.time())
                    logger.debug(f"Model {model_name} retrieved from cache")
                    return model

            # Load model outside of cache lock but inside model lock
            try:
                logger.info(f"Loading model {model_name}...")
                model = self._load_model_sync(model_name)

                if model is not None:
                    # Estimate model size
                    model_size = self._estimate_model_size(model_name)

                    # Check if we can fit this model in GPU memory
                    if not self._gpu_manager.can_allocate_model(model_size):
                        logger.warning(f"Insufficient GPU memory for model {model_name}")
                        # Try to free up space
                        self._evict_models_for_space(model_size, self._gpu_manager)

                    # Add to cache (with cache lock)
                    with self._cache_lock:
                        # Enforce cache size limit
                        if len(self._models) >= self._max_cache_size:
                            # Remove least recently used model
                            lru_model = min(self._models.items(), key=lambda x: x[1][1])
                            del self._models[lru_model[0]]
                            if lru_model[0] in self._model_sizes:
                                del self._model_sizes[lru_model[0]]
                            logger.info(f"Evicted model {lru_model[0]} from cache (LRU)")

                        self._models[model_name] = (model, time.time())
                        self._model_sizes[model_name] = model_size
                        logger.info(f"Model {model_name} added to cache")

                return model

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise ParakeetModelError(f"Failed to load model {model_name}: {e}")

    async def get_model_async(self, model_name: str, force_reload: bool = False) -> Optional[Any]:
        """Async wrapper for get_model.

        Args:
            model_name: Name of the Parakeet model
            force_reload: Force reload even if cached

        Returns:
            Loaded model or None if loading fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_model, model_name, force_reload)

    def _load_model_sync(self, model_name: str) -> Any:
        """Synchronously load a Parakeet model.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model
        """
        if not self._is_valid_model_name(model_name):
            raise ValueError(f"Invalid model name: {model_name}")

        try:
            if not _ensure_nemo():
                raise RuntimeError("NeMo not available")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name)
            if TORCH_AVAILABLE and self._gpu_manager.device != "cpu":
                model = model.to(self._gpu_manager.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def _is_valid_model_name(self, model_name: str) -> bool:
        """Validate model name format.

        Args:
            model_name: Model name to validate

        Returns:
            True if valid
        """
        # Add validation logic for Parakeet model names
        # This is a basic check - expand as needed
        return bool(model_name and isinstance(model_name, str))

    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in bytes.

        Args:
            model_name: Name of the model

        Returns:
            Estimated size in bytes
        """
        # Rough estimates for common Parakeet models
        size_map = {
            "stt_en_fastconformer_transducer_large": 600 * 1024 * 1024,  # 600MB
            "stt_en_conformer_transducer_large": 500 * 1024 * 1024,  # 500MB
            "stt_en_conformer_transducer_medium": 300 * 1024 * 1024,  # 300MB
            "stt_en_conformer_transducer_small": 150 * 1024 * 1024,  # 150MB
        }

        # Default to 400MB for unknown models
        return size_map.get(model_name, 400 * 1024 * 1024)

    def _evict_models_for_space(self, required_size: int, gpu_manager: GPUManager) -> None:
        """Evict models to free up space.

        Args:
            required_size: Required size in bytes
            gpu_manager: GPU manager instance
        """
        with self._cache_lock:
            if not self._models:
                return

            # Sort models by last used time (oldest first)
            sorted_models = sorted(self._models.items(), key=lambda x: x[1][1])

            freed_space = 0
            models_to_evict = []

            for model_name, (_model, _last_used) in sorted_models:
                if freed_space >= required_size:
                    break

                model_size = self._model_sizes.get(model_name, 0)
                models_to_evict.append(model_name)
                freed_space += model_size

            # Evict models
            for model_name in models_to_evict:
                del self._models[model_name]
                if model_name in self._model_sizes:
                    del self._model_sizes[model_name]
                logger.info(f"Evicted model {model_name} to free GPU memory")

            # Clean up GPU memory
            if models_to_evict:
                gpu_manager.cleanup_gpu_memory()

    def clear_cache(self) -> None:
        """Clear all cached models."""
        with self._cache_lock:
            self._models.clear()
            self._model_sizes.clear()
            logger.info("Model cache cleared")

            # Clean up GPU memory
            if hasattr(self, "_gpu_manager"):
                self._gpu_manager.cleanup_gpu_memory()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._cache_lock:
            total_size = sum(self._model_sizes.values())
            return {
                "cached_models": len(self._models),
                "model_names": list(self._models.keys()),
                "total_size_mb": total_size / (1024 * 1024),
                "max_cache_size": self._max_cache_size,
                "device": self._gpu_manager.device if hasattr(self, "_gpu_manager") else "unknown",
            }
