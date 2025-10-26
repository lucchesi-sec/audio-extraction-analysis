"""Comprehensive caching system for transcription results."""
from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Extracted helpers
from .eviction import (
    select_fifo_victim,
    select_lfu_victim,
    select_lru_victim,
    select_size_victim,
    select_ttl_victim,
)
from .compression import compress_value, decompress_value


class CachePolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"  # Random eviction


@dataclass
class CacheKey:
    """Content-based cache key."""

    file_hash: str
    provider: str
    settings_hash: str

    # Class-level file hash cache keyed by (path, mtime, size) to eliminate redundant I/O
    # Shared across all instances to cache file hashes based on file metadata
    _file_hash_cache: ClassVar[Dict[Tuple[str, float, int], str]] = {}

    def __str__(self) -> str:
        """String representation of cache key."""
        return f"{self.file_hash}:{self.provider}:{self.settings_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_hash": self.file_hash,
            "provider": self.provider,
            "settings_hash": self.settings_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheKey":
        """Create from dictionary."""
        return cls(
            file_hash=data["file_hash"],
            provider=data["provider"],
            settings_hash=data["settings_hash"],
        )

    @classmethod
    def from_file(cls, file_path: Path, provider: str, settings: Dict[str, Any]) -> "CacheKey":
        """Generate cache key from file and settings.

        Args:
            file_path: Path to audio file
            provider: Transcription provider
            settings: Provider settings

        Returns:
            CacheKey instance
        """
        # Hash file content
        file_hash = cls._hash_file(file_path)

        # Hash settings
        settings_str = json.dumps(settings, sort_keys=True)
        settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()[:16]

        return cls(file_hash=file_hash, provider=provider, settings_hash=settings_hash)

    @classmethod
    def _hash_file(cls, file_path: Path, chunk_size: int = 8192) -> str:
        """Generate SHA256 hash of file content with intelligent caching.

        Uses file metadata (path, mtime, size) as cache key to avoid redundant I/O.
        For a 2GB file, this reduces 260k+ chunk reads to zero on cache hit.

        Args:
            file_path: Path to file
            chunk_size: Chunk size for reading

        Returns:
            File hash string
        """
        import os

        # Get file stats for cache key
        stat = os.stat(file_path)
        cache_key = (str(file_path), stat.st_mtime, stat.st_size)

        # Check cache first
        if cache_key in cls._file_hash_cache:
            return cls._file_hash_cache[cache_key]

        # Cache miss - compute hash
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)

        file_hash = sha256.hexdigest()[:32]

        # Store in cache
        cls._file_hash_cache[cache_key] = file_hash

        return file_hash

    @classmethod
    def clear_hash_cache(cls, file_path: Optional[Path] = None) -> int:
        """Clear file hash cache entries.

        Args:
            file_path: Specific file to clear (None for all)

        Returns:
            Number of entries cleared
        """
        if file_path is None:
            # Clear all entries
            count = len(cls._file_hash_cache)
            cls._file_hash_cache.clear()
            return count

        # Clear entries for specific file path
        path_str = str(file_path)
        keys_to_remove = [key for key in cls._file_hash_cache.keys() if key[0] == path_str]
        for key in keys_to_remove:
            del cls._file_hash_cache[key]

        return len(keys_to_remove)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: CacheKey
    value: Any
    size: int
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if expired
        """
        if self.ttl is None:
            return False

        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def age_seconds(self) -> float:
        """Get age in seconds.

        Returns:
            Age in seconds
        """
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Handle different value types safely
        value_dict = None
        if hasattr(self.value, 'to_dict') and callable(getattr(self.value, 'to_dict')):
            value_dict = self.value.to_dict()
        else:
            # For simple types that are JSON serializable
            try:
                import json
                json.dumps(self.value)  # Test if serializable
                value_dict = self.value
            except (TypeError, ValueError):
                # If not serializable, store as string representation
                value_dict = str(self.value)
        
        return {
            "key": self.key.to_dict(),
            "value": value_dict,
            "value_type": type(self.value).__name__,
            "size": self.size,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        # Reconstruct the cache key
        cache_key = CacheKey.from_dict(data["key"])
        
        # Reconstruct the value based on its type
        value = data["value"]
        value_type = data.get("value_type", "dict")
        
        if value_type == "TranscriptionResult" and isinstance(value, dict):
            # Try importing TranscriptionResult with fallback handling
            try:
                from ..models.transcription import TranscriptionResult
            except (ImportError, ValueError):
                try:
                    from models.transcription import TranscriptionResult
                except ImportError:
                    # If TranscriptionResult not available, keep as dict
                    value = value
                else:
                    value = TranscriptionResult.from_dict(value)
            else:
                value = TranscriptionResult.from_dict(value)
        # For other types, use the value as-is (assuming it's JSON-serializable)
        
        return cls(
            key=cache_key,
            value=value,
            size=data["size"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data["access_count"],
            ttl=data["ttl"],
            metadata=data["metadata"],
        )


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate.

        Returns:
            Hit rate percentage
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Stats dictionary
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2f}%",
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024),
            "entry_count": self.entry_count,
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        pass

    @abstractmethod
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry in cache.

        Args:
            key: Cache key
            entry: Cache entry

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Get cache size in bytes.

        Returns:
            Size in bytes
        """
        pass

    @abstractmethod
    def keys(self) -> Set[str]:
        """Get all cache keys.

        Returns:
            Set of keys
        """
        pass


class TranscriptionCache:
    """Main transcription cache with multiple backends."""

    def __init__(
        self,
        backends: Optional[List[CacheBackend]] = None,
        policy: CachePolicy = CachePolicy.LRU,
        max_size_mb: int = 1000,
        max_entries: int = 10000,
        default_ttl: Optional[int] = 3600,
        enable_compression: bool = True,
        enable_warming: bool = False,
    ):
        """Initialize transcription cache.

        Args:
            backends: List of cache backends (hierarchical)
            policy: Eviction policy
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            default_ttl: Default TTL in seconds
            enable_compression: Enable value compression
            enable_warming: Enable cache warming
        """
        if backends is None:
            # Import here to avoid circular import
            from .backends import InMemoryCache
            self.backends = [InMemoryCache()]
        else:
            self.backends = backends
        self.policy = policy
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.enable_warming = enable_warming

        self.stats = CacheStats()
        self._lock = RLock()
        self._warm_keys: Set[str] = set()

        logger.info(
            f"Initialized TranscriptionCache with {len(self.backends)} backend(s), "
            f"policy={policy.value}, max_size={max_size_mb}MB"
        )

    def get(self, file_path: Path, provider: str, settings: Dict[str, Any]) -> Optional[Any]:
        """Get transcription from cache.

        Args:
            file_path: Audio file path
            provider: Provider name
            settings: Provider settings

        Returns:
            Cached transcription or None
        """
        # Generate cache key
        cache_key = CacheKey.from_file(file_path, provider, settings)
        key_str = str(cache_key)

        with self._lock:
            # Try each backend in order
            for i, backend in enumerate(self.backends):
                entry = backend.get(key_str)

                if entry:
                    # Check expiration
                    if entry.is_expired():
                        backend.delete(key_str)
                        continue

                    # Update stats
                    self.stats.hits += 1
                    entry.touch()

                    # Promote to higher cache levels
                    if i > 0:
                        self._promote_entry(key_str, entry, i)

                    # Decompress if needed
                    value = (
                        self._decompress(entry.value) if self.enable_compression else entry.value
                    )

                    logger.debug(f"Cache hit for {key_str} from backend {i}")
                    return value

            # Cache miss
            self.stats.misses += 1
            logger.debug(f"Cache miss for {key_str}")
            return None

    def put(
        self,
        file_path: Path,
        provider: str,
        settings: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Put transcription in cache.

        Args:
            file_path: Audio file path
            provider: Provider name
            settings: Provider settings
            value: Transcription result
            ttl: TTL in seconds
            metadata: Additional metadata

        Returns:
            True if cached successfully
        """
        # Generate cache key
        cache_key = CacheKey.from_file(file_path, provider, settings)
        key_str = str(cache_key)

        # Compress if enabled
        cached_value = self._compress(value) if self.enable_compression else value

        # Calculate size
        size = self._calculate_size(cached_value)

        # Check size limits
        if size > self.max_size_bytes:
            logger.warning(f"Value too large to cache: {size} > {self.max_size_bytes}")
            return False

        # Create entry
        entry = CacheEntry(
            key=cache_key,
            value=cached_value,
            size=size,
            ttl=ttl or self.default_ttl,
            metadata=metadata or {},
        )

        with self._lock:
            # Evict if necessary
            self._evict_if_needed(size)

            # Store in primary backend
            success = self.backends[0].put(key_str, entry)

            if success:
                self.stats.entry_count += 1
                self.stats.size_bytes += size
                logger.debug(f"Cached {key_str} ({size} bytes)")

            return success

    def invalidate(self, file_path: Optional[Path] = None, provider: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            file_path: File to invalidate (None for all)
            provider: Provider to invalidate (None for all)

        Returns:
            Number of entries invalidated
        """
        count = 0

        with self._lock:
            keys_to_delete = []

            for backend in self.backends:
                for key_str in backend.keys():
                    # Parse key
                    parts = key_str.split(":")
                    if len(parts) != 3:
                        continue

                    file_hash, key_provider, _ = parts

                    # Check match criteria
                    if provider and key_provider != provider:
                        continue

                    if file_path:
                        # Need to match file hash
                        test_hash = CacheKey._hash_file(file_path)
                        if file_hash != test_hash[:32]:
                            continue

                    keys_to_delete.append(key_str)

            # Delete matching keys
            for key_str in keys_to_delete:
                for backend in self.backends:
                    if backend.delete(key_str):
                        count += 1
                        self.stats.entry_count -= 1

            # Clear file hash cache for invalidated file
            if file_path:
                CacheKey.clear_hash_cache(file_path)
            elif count > 0:
                # If invalidating all, clear entire hash cache
                CacheKey.clear_hash_cache()

        logger.info(f"Invalidated {count} cache entries")
        return count

    def warm(self, entries: List[Tuple[Path, str, Dict[str, Any], Any]]) -> int:
        """Warm cache with pre-computed entries.

        Args:
            entries: List of (file_path, provider, settings, value) tuples

        Returns:
            Number of entries warmed
        """
        if not self.enable_warming:
            logger.warning("Cache warming is disabled")
            return 0

        count = 0
        for file_path, provider, settings, value in entries:
            if self.put(file_path, provider, settings, value):
                key = str(CacheKey.from_file(file_path, provider, settings))
                self._warm_keys.add(key)
                count += 1

        logger.info(f"Warmed cache with {count} entries")
        return count

    def _evict_if_needed(self, required_size: int):
        """Evict entries if needed to make space.

        Args:
            required_size: Size needed in bytes
        """
        # Check entry count limit
        while self.stats.entry_count >= self.max_entries:
            self._evict_one()

        # Check size limit
        while self.stats.size_bytes + required_size > self.max_size_bytes:
            if not self._evict_one():
                break

    def _evict_one(self) -> bool:
        """Evict one entry based on policy.

        Returns:
            True if evicted
        """
        backend = self.backends[0]
        keys = backend.keys()

        if not keys:
            return False

        # Select victim based on policy
        if self.policy == CachePolicy.LRU:
            victim_key = self._select_lru_victim(backend, keys)
        elif self.policy == CachePolicy.LFU:
            victim_key = self._select_lfu_victim(backend, keys)
        elif self.policy == CachePolicy.TTL:
            victim_key = self._select_ttl_victim(backend, keys)
        elif self.policy == CachePolicy.SIZE:
            victim_key = self._select_size_victim(backend, keys)
        elif self.policy == CachePolicy.FIFO:
            victim_key = self._select_fifo_victim(backend, keys)
        else:  # RANDOM
            import random

            victim_key = random.choice(list(keys))

        # Skip warm entries if possible
        if victim_key in self._warm_keys and len(keys) > len(self._warm_keys):
            # Try to find non-warm victim
            for key in keys:
                if key not in self._warm_keys:
                    victim_key = key
                    break

        # Evict victim
        entry = backend.get(victim_key)
        if entry and backend.delete(victim_key):
            self.stats.evictions += 1
            self.stats.entry_count -= 1
            self.stats.size_bytes -= entry.size
            logger.debug(f"Evicted {victim_key}")
            return True

        return False

    def _select_lru_victim(self, backend: CacheBackend, keys: Set[str]) -> str:
        """Select LRU victim.

        Args:
            backend: Cache backend
            keys: Available keys

        Returns:
            Victim key
        """
        return select_lru_victim(backend, keys)

    def _select_lfu_victim(self, backend: CacheBackend, keys: Set[str]) -> str:
        """Select LFU victim.

        Args:
            backend: Cache backend
            keys: Available keys

        Returns:
            Victim key
        """
        return select_lfu_victim(backend, keys)

    def _select_ttl_victim(self, backend: CacheBackend, keys: Set[str]) -> str:
        """Select TTL victim (closest to expiration).

        Args:
            backend: Cache backend
            keys: Available keys

        Returns:
            Victim key
        """
        return select_ttl_victim(backend, keys)

    def _select_size_victim(self, backend: CacheBackend, keys: Set[str]) -> str:
        """Select largest entry as victim.

        Args:
            backend: Cache backend
            keys: Available keys

        Returns:
            Victim key
        """
        return select_size_victim(backend, keys)

    def _select_fifo_victim(self, backend: CacheBackend, keys: Set[str]) -> str:
        """Select FIFO victim (oldest creation time).

        Args:
            backend: Cache backend
            keys: Available keys

        Returns:
            Victim key
        """
        return select_fifo_victim(backend, keys)

    def _promote_entry(self, key: str, entry: CacheEntry, from_level: int):
        """Promote entry to higher cache levels.

        Args:
            key: Cache key
            entry: Cache entry
            from_level: Current backend level
        """
        # Promote to all higher levels
        for i in range(from_level):
            self.backends[i].put(key, entry)

    def _compress(self, value: Any) -> bytes:
        """Compress value for storage using safe JSON serialization."""
        return compress_value(value)

    def _decompress(self, data: bytes) -> Any:
        """Decompress value from storage using safe JSON deserialization."""
        value = decompress_value(data)
        if value is None:
            logger.error("Failed to decompress cache value")
        return value

    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes.

        Args:
            value: Value to measure

        Returns:
            Size in bytes
        """
        import sys

        if isinstance(value, bytes):
            return len(value)

        # Rough estimate for other types
        return sys.getsizeof(value)

    def _prepare_cache_value(self, value: Any) -> Any:
        """Prepare value for caching by applying compression if enabled.

        Args:
            value: Original value to cache

        Returns:
            Compressed or original value
        """
        return self._compress(value) if self.enable_compression else value

    def _validate_entry_size(self, size: int) -> bool:
        """Validate that entry size is within cache limits.

        Args:
            size: Entry size in bytes

        Returns:
            True if size is acceptable, False otherwise
        """
        if size > self.max_size_bytes:
            logger.warning(f"Value too large to cache: {size} > {self.max_size_bytes}")
            return False
        return True

    def _create_cache_entry(
        self,
        cache_key: CacheKey,
        cached_value: Any,
        size: int,
        ttl: Optional[int],
        metadata: Optional[Dict[str, Any]],
    ) -> CacheEntry:
        """Create cache entry with metadata.

        Args:
            cache_key: Cache key
            cached_value: Prepared cache value
            size: Entry size in bytes
            ttl: TTL in seconds
            metadata: Additional metadata

        Returns:
            CacheEntry instance
        """
        return CacheEntry(
            key=cache_key,
            value=cached_value,
            size=size,
            ttl=ttl or self.default_ttl,
            metadata=metadata or {},
        )

    def _store_entry_in_backend(self, key_str: str, entry: CacheEntry, size: int) -> bool:
        """Store entry in backend with eviction and stats tracking.

        Args:
            key_str: String representation of cache key
            entry: Cache entry to store
            size: Entry size in bytes

        Returns:
            True if stored successfully
        """
        with self._lock:
            # Evict if necessary
            self._evict_if_needed(size)

            # Store in primary backend
            success = self.backends[0].put(key_str, entry)

            if success:
                self.stats.entry_count += 1
                self.stats.size_bytes += size
                logger.debug(f"Cached {key_str} ({size} bytes)")

            return success

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        return self.stats

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0

        with self._lock:
            for backend in self.backends:
                count += backend.clear()

            self.stats = CacheStats()
            self._warm_keys.clear()

        logger.info(f"Cleared {count} cache entries")
        return count


# Backend implementations imported lazily to avoid circular imports
