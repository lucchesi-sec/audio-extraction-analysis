"""Comprehensive caching system for transcription results.

This module provides a multi-level caching system for transcription results with:
- Content-based cache keys (file hash + provider + settings)
- Multiple eviction policies (LRU, LFU, TTL, SIZE, FIFO, RANDOM)
- Hierarchical backend support (in-memory, disk, distributed)
- Optional compression for storage efficiency
- Cache warming for pre-loading frequently used data
- Thread-safe operations with fine-grained locking

Architecture:
    TranscriptionCache (main interface)
    ├── CacheBackend[] (hierarchical backends, L1->L2->L3)
    ├── CacheKey (content-based addressing)
    ├── CacheEntry (value + metadata + stats)
    └── CacheStats (hit/miss/eviction metrics)

Example:
    ```python
    from pathlib import Path
    cache = TranscriptionCache(
        policy=CachePolicy.LRU,
        max_size_mb=500,
        enable_compression=True
    )

    # Check cache
    result = cache.get(Path("audio.mp3"), "whisper", {"model": "base"})
    if result is None:
        result = transcribe_audio(...)
        cache.put(Path("audio.mp3"), "whisper", {"model": "base"}, result)
    ```
"""
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
    """Cache eviction policies.

    Defines strategies for selecting which cache entries to remove when space is needed:

    - LRU: Best for workloads with temporal locality (recent files accessed again)
    - LFU: Best for workloads with frequency patterns (popular files accessed repeatedly)
    - TTL: Best when data freshness matters (expire old transcriptions)
    - SIZE: Best for optimizing storage efficiency (remove largest items first)
    - FIFO: Simple queue-based eviction (predictable, no access tracking overhead)
    - RANDOM: Fastest eviction (no metadata tracking, minimal overhead)
    """

    LRU = "lru"  # Least Recently Used - evict oldest access time
    LFU = "lfu"  # Least Frequently Used - evict lowest access count
    TTL = "ttl"  # Time To Live - evict closest to expiration
    SIZE = "size"  # Size-based eviction - evict largest entries
    FIFO = "fifo"  # First In First Out - evict oldest creation time
    RANDOM = "random"  # Random eviction - no priority logic


@dataclass
class CacheKey:
    """Content-based cache key for deterministic transcription result lookup.

    Uses a composite key of (file_hash, provider, settings_hash) to ensure:
    - Same file + provider + settings → same cache key (cache hit)
    - Any change in file content, provider, or settings → different key (cache miss)

    This prevents serving stale results when:
    - Audio file is modified (detected via content hash)
    - Provider is changed (e.g., whisper → google)
    - Settings are changed (e.g., language, model, quality)

    Attributes:
        file_hash: SHA256 hash of file content (first 32 chars)
        provider: Transcription provider name (e.g., 'whisper', 'google')
        settings_hash: SHA256 hash of provider settings dict (first 16 chars)
    """

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
    """Cache entry with metadata for tracking usage and expiration.

    Stores a cached value along with metadata needed for:
    - Eviction policy decisions (access time, access count, age, size)
    - Expiration handling (TTL)
    - Cache analytics (metadata dict for custom tracking)

    Attributes:
        key: Content-based cache key
        value: Cached transcription result (may be compressed bytes)
        size: Entry size in bytes (for size-based eviction)
        created_at: Entry creation timestamp (for FIFO, TTL policies)
        accessed_at: Last access timestamp (for LRU policy)
        access_count: Number of times accessed (for LFU policy)
        ttl: Time-to-live in seconds (None = no expiration)
        metadata: Additional metadata dict for custom tracking
    """

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
        """Convert to dictionary for JSON serialization.

        Handles multiple value types with fallback strategy:
        1. Custom objects with to_dict() method → call to_dict()
        2. JSON-serializable types (dict, list, str, int, etc.) → use as-is
        3. Non-serializable types (bytes, custom objects) → convert to string

        This allows safe serialization of compressed values, TranscriptionResult objects,
        and other complex types without losing critical data.

        Returns:
            Dictionary representation suitable for JSON persistence
        """
        # Handle different value types safely
        value_dict = None
        if hasattr(self.value, 'to_dict') and callable(getattr(self.value, 'to_dict')):
            # Custom objects with serialization support
            value_dict = self.value.to_dict()
        else:
            # For simple types that are JSON serializable
            try:
                import json
                json.dumps(self.value)  # Test if serializable
                value_dict = self.value
            except (TypeError, ValueError):
                # If not serializable (e.g., bytes, custom objects), store as string
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
        """Create cache entry from dictionary representation.

        Reconstructs a CacheEntry from JSON data with intelligent type restoration:
        - Deserializes the cache key
        - Restores the original value type (TranscriptionResult, dict, etc.)
        - Uses fallback import logic to handle different module paths

        The import fallback handles cases where:
        1. Cache was created in a different execution context (tests vs. main app)
        2. Module structure changed between cache write and read
        3. TranscriptionResult class is not available (returns raw dict)

        Args:
            data: Dictionary containing serialized entry data

        Returns:
            Reconstructed CacheEntry instance
        """
        # Reconstruct the cache key
        cache_key = CacheKey.from_dict(data["key"])

        # Reconstruct the value based on its type
        value = data["value"]
        value_type = data.get("value_type", "dict")

        if value_type == "TranscriptionResult" and isinstance(value, dict):
            # Try importing TranscriptionResult with multiple fallback paths
            # to handle different execution contexts (tests, main app, etc.)
            try:
                from ..models.transcription import TranscriptionResult
            except (ImportError, ValueError):
                try:
                    from models.transcription import TranscriptionResult
                except ImportError:
                    # If TranscriptionResult not available, keep as dict (graceful degradation)
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
    """Abstract base class for cache backends.

    Defines the interface for cache storage implementations. Backends can be:
    - In-memory: Fast L1 cache using dict/OrderedDict
    - Disk-based: Persistent L2 cache using SQLite or file system
    - Distributed: Network L3 cache using Redis or Memcached

    Multiple backends can be composed hierarchically (L1 → L2 → L3) where:
    - get() checks backends in order (L1 first, L2 fallback, etc.)
    - put() writes to primary backend (L1)
    - Cache hits promote entries to higher levels (L2 → L1)

    Implementations must be thread-safe if used in multi-threaded contexts.
    """

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
    """Main transcription cache with multiple backends and advanced features.

    A production-ready caching system that provides:
    - Content-based addressing (cache invalidation on file changes)
    - Hierarchical backend support (L1/L2/L3 with automatic promotion)
    - Multiple eviction policies (LRU, LFU, TTL, SIZE, FIFO, RANDOM)
    - Optional compression (reduces memory footprint for large transcriptions)
    - Cache warming (pre-populate with frequently accessed data)
    - Thread-safe operations (safe for multi-threaded transcription pipelines)

    Cache Hit Flow:
        1. Generate cache key from (file_hash, provider, settings_hash)
        2. Query backends in order (L1 → L2 → L3)
        3. Check expiration, promote to higher levels if needed
        4. Decompress value if compression enabled
        5. Update access stats and return result

    Cache Miss Flow:
        1. Transcribe audio (external operation)
        2. Compress result if enabled
        3. Evict entries if cache full (based on policy)
        4. Store in primary backend (L1)
        5. Update size/count stats

    Thread Safety:
        All operations protected by RLock for safe concurrent access.
        Stats updates and eviction decisions are atomic.
    """

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

        # Prepare value (compress if enabled)
        cached_value = self._prepare_cache_value(value)

        # Calculate and validate size
        size = self._calculate_size(cached_value)
        if not self._validate_entry_size(size):
            return False

        # Create cache entry
        entry = self._create_cache_entry(cache_key, cached_value, size, ttl, metadata)

        # Store in backend
        return self._store_entry_in_backend(key_str, entry, size)

    def invalidate(self, file_path: Optional[Path] = None, provider: Optional[str] = None) -> int:
        """Invalidate cache entries matching the specified criteria.

        Removes cached transcriptions based on file path and/or provider filters:
        - invalidate(None, None) → clear entire cache
        - invalidate(file_path, None) → clear all providers for this file
        - invalidate(None, provider) → clear all files for this provider
        - invalidate(file_path, provider) → clear specific file+provider combo

        Also clears the file hash cache to ensure recomputation on next access.

        Use cases:
            - File modified → invalidate(file_path) to force re-transcription
            - Provider settings changed → invalidate(provider=name) to clear old results
            - Cache corruption → invalidate() to clear everything

        Args:
            file_path: Specific file to invalidate (None = all files)
            provider: Specific provider to invalidate (None = all providers)

        Returns:
            Number of cache entries removed
        """
        count = 0

        with self._lock:
            keys_to_delete = []

            # Phase 1: Identify matching keys
            for backend in self.backends:
                for key_str in backend.keys():
                    # Parse key format: "file_hash:provider:settings_hash"
                    parts = key_str.split(":")
                    if len(parts) != 3:
                        continue  # Skip malformed keys

                    file_hash, key_provider, _ = parts

                    # Apply provider filter
                    if provider and key_provider != provider:
                        continue

                    # Apply file path filter (requires hash recomputation)
                    if file_path:
                        test_hash = CacheKey._hash_file(file_path)
                        if file_hash != test_hash[:32]:
                            continue

                    keys_to_delete.append(key_str)

            # Phase 2: Delete matching keys from all backends
            for key_str in keys_to_delete:
                for backend in self.backends:
                    if backend.delete(key_str):
                        count += 1
                        self.stats.entry_count -= 1

            # Phase 3: Clear file hash cache to ensure freshness
            if file_path:
                CacheKey.clear_hash_cache(file_path)
            elif count > 0:
                # If invalidating all, clear entire hash cache
                CacheKey.clear_hash_cache()

        logger.info(f"Invalidated {count} cache entries")
        return count

    def warm(self, entries: List[Tuple[Path, str, Dict[str, Any], Any]]) -> int:
        """Pre-populate cache with frequently accessed transcription results.

        Cache warming improves performance by pre-loading data before it's requested:
        - Reduces cold-start latency for known frequent queries
        - Protects warm entries from eviction (see _evict_one)
        - Useful for batch processing workflows with predictable access patterns

        Warm entries are tracked in self._warm_keys and given eviction protection:
        when a warm entry is selected for eviction, the eviction algorithm will
        try to find a non-warm alternative if available.

        Must be enabled via enable_warming=True during initialization.

        Args:
            entries: List of (file_path, provider, settings, result) tuples to pre-cache

        Returns:
            Number of entries successfully warmed (may be less than input if cache full)
        """
        if not self.enable_warming:
            logger.warning("Cache warming is disabled")
            return 0

        count = 0
        for file_path, provider, settings, value in entries:
            if self.put(file_path, provider, settings, value):
                # Track as warm entry for eviction protection
                key = str(CacheKey.from_file(file_path, provider, settings))
                self._warm_keys.add(key)
                count += 1

        logger.info(f"Warmed cache with {count} entries")
        return count

    def _evict_if_needed(self, required_size: int):
        """Evict entries if needed to make space for new entry.

        Performs eviction in two phases:
        1. Entry count limit: Evict until entry_count < max_entries
        2. Size limit: Evict until size_bytes + required_size <= max_size_bytes

        This ensures both constraints are satisfied before inserting the new entry.
        Eviction policy (LRU, LFU, etc.) determines which entries are removed.

        Args:
            required_size: Size needed in bytes for new entry
        """
        # Check entry count limit
        while self.stats.entry_count >= self.max_entries:
            self._evict_one()

        # Check size limit
        while self.stats.size_bytes + required_size > self.max_size_bytes:
            if not self._evict_one():
                break  # No more entries to evict

    def _evict_one(self) -> bool:
        """Evict one entry based on configured eviction policy.

        Selects a victim entry using the configured policy (LRU, LFU, TTL, etc.)
        and removes it from the primary backend. Warm entries (pre-loaded via warm())
        are protected from eviction when possible.

        Eviction process:
        1. Select victim based on policy (delegate to policy-specific selector)
        2. Protect warm entries (skip if non-warm alternatives exist)
        3. Delete from backend and update stats

        Returns:
            True if an entry was evicted, False if cache is empty
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

        # Protect warm entries from eviction if non-warm alternatives exist
        if victim_key in self._warm_keys and len(keys) > len(self._warm_keys):
            # Try to find non-warm victim (simple linear search)
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
        """Promote entry to higher (faster) cache levels after a cache hit.

        When an entry is found in a lower-level backend (e.g., L2 disk cache),
        it's copied to all higher levels (e.g., L1 memory cache) for faster
        subsequent access. This implements a cache hierarchy optimization.

        Example:
            Entry found in L2 (disk) → copy to L1 (memory)
            Next access hits L1 directly (much faster)

        Args:
            key: Cache key string
            entry: Cache entry to promote
            from_level: Backend index where entry was found (0=L1, 1=L2, etc.)
        """
        # Promote to all higher levels (0 to from_level-1)
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

        Part of the put() operation pipeline. Compresses the transcription result
        if compression is enabled, reducing memory/disk usage at the cost of
        CPU cycles for compression/decompression.

        Args:
            value: Original transcription result to cache

        Returns:
            Compressed bytes if compression enabled, otherwise original value
        """
        return self._compress(value) if self.enable_compression else value

    def _validate_entry_size(self, size: int) -> bool:
        """Validate that entry size is within cache limits.

        Part of the put() operation pipeline. Rejects entries that are larger
        than the entire cache size (would never fit even after eviction).

        Args:
            size: Entry size in bytes (after compression if enabled)

        Returns:
            True if size is acceptable, False if entry is too large to cache
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

        Part of the put() operation pipeline. Constructs a CacheEntry with
        all necessary metadata for eviction policy decisions and analytics.

        Args:
            cache_key: Content-based cache key
            cached_value: Prepared cache value (possibly compressed)
            size: Entry size in bytes
            ttl: Time-to-live in seconds (None uses default_ttl)
            metadata: Additional metadata dict for custom tracking

        Returns:
            Fully constructed CacheEntry instance
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

        Final step of the put() operation pipeline. Evicts entries if necessary
        to make space, stores the entry in the primary backend, and updates
        cache statistics atomically.

        Thread Safety:
            Protected by self._lock for atomic eviction + insertion + stats update.

        Args:
            key_str: String representation of cache key
            entry: Cache entry to store
            size: Entry size in bytes (for stats tracking)

        Returns:
            True if stored successfully, False on backend error
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
#
# The TranscriptionCache class accepts a list of CacheBackend implementations,
# but the concrete backend classes (InMemoryCache, DiskCache, etc.) may import
# from this module. To break the circular dependency, backends are imported
# lazily within methods (e.g., __init__) rather than at module level.
#
# Example usage:
#     from .backends import InMemoryCache, DiskCache
#     cache = TranscriptionCache(backends=[InMemoryCache(), DiskCache()])
#
# The default behavior (backends=None) lazily imports InMemoryCache to provide
# a working cache without requiring explicit backend configuration.
