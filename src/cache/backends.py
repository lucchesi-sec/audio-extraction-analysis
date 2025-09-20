"""Cache backend implementations."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock, RLock
from typing import Optional, Set, Union

from .common import CacheUtils, SizeLimitManager
from .transcription_cache import CacheBackend, CacheEntry

logger = logging.getLogger(__name__)


class InMemoryCache(CacheBackend):
    """In-memory cache backend with OrderedDict."""

    def __init__(self, max_size_mb: int = 100):
        """Initialize in-memory cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._size_manager = SizeLimitManager(self.max_size_bytes)

        logger.info(f"Initialized InMemoryCache with max_size={max_size_mb}MB")

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from memory.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                # Move to end (LRU)
                self._cache.move_to_end(key)
            return entry

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry in memory.

        Args:
            key: Cache key
            entry: Cache entry

        Returns:
            True if successful
        """
        # Normalize key for consistency
        normalized_key = CacheUtils.normalize_key(key)
        
        with self._lock:
            # Check if entry can fit at all
            if not self._size_manager.can_fit(entry.size):
                return False

            # Remove old entry if exists
            if normalized_key in self._cache:
                old_entry = self._cache[normalized_key]
                self._size_manager.remove_entry(old_entry.size)

            # Evict if needed to make space
            while self._size_manager.would_exceed_limit(entry.size):
                if not self._evict_oldest():
                    return False

            # Add new entry
            self._cache[normalized_key] = entry
            self._size_manager.add_entry(entry.size)
            return True

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            return key in self._cache

    def delete(self, key: str) -> bool:
        """Delete entry from memory.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._size_manager.remove_entry(entry.size)
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._size_manager.reset()
            return count

    def size(self) -> int:
        """Get cache size in bytes.

        Returns:
            Size in bytes
        """
        return self._size_manager.current_size

    def keys(self) -> Set[str]:
        """Get all cache keys.

        Returns:
            Set of keys
        """
        with self._lock:
            return set(self._cache.keys())

    def _evict_oldest(self) -> bool:
        """Evict oldest entry.

        Returns:
            True if evicted
        """
        if not self._cache:
            return False

        # Get oldest (first) entry
        key, entry = next(iter(self._cache.items()))
        self._size_manager.remove_entry(entry.size)
        del self._cache[key]
        return True


class DiskCache(CacheBackend):
    """Disk-based cache backend using SQLite."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, max_size_mb: int = 1000):
        """Initialize disk cache.

        Args:
            cache_dir: Cache directory (string or Path object)
            max_size_mb: Maximum cache size in MB
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".audio_extraction" / "cache"
        else:
            self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = Lock()

        self._init_database()
        logger.info(f"Initialized DiskCache at {self.cache_dir} with max_size={max_size_mb}MB")

    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl INTEGER,
                    metadata TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_accessed 
                ON cache_entries(accessed_at)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_size 
                ON cache_entries(size)
            """
            )

            conn.commit()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        SELECT value, size, created_at, accessed_at, 
                               access_count, ttl, metadata
                        FROM cache_entries
                        WHERE key = ?
                    """,
                        (key,),
                    )

                    row = cursor.fetchone()
                    if row:
                        # Update access time and count
                        cursor.execute(
                            """
                            UPDATE cache_entries 
                            SET accessed_at = ?, access_count = access_count + 1
                            WHERE key = ?
                        """,
                            (time.time(), key),
                        )
                        conn.commit()

                        # Deserialize entry using safe JSON
                        try:
                            entry_dict = json.loads(row[0].decode('utf-8'))
                            # Import CacheEntry - handle circular import properly
                            try:
                                from .transcription_cache import CacheEntry
                            except ImportError:
                                # If circular import, use the already imported class
                                CacheEntry = globals().get('CacheEntry')
                                if not CacheEntry:
                                    raise ImportError("CacheEntry not available")
                            entry_data = CacheEntry.from_dict(entry_dict)
                            return entry_data
                        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ImportError) as e:
                            logger.error(f"Failed to deserialize cache entry: {e}")
                            # Clean up corrupted entry
                            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                            conn.commit()
                            return None

                    return None

            except Exception as e:
                logger.error(f"Failed to get from disk cache: {e}")
                return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry on disk.

        Args:
            key: Cache key
            entry: Cache entry

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Check size
                if entry.size > self.max_size_bytes:
                    return False

                # Evict if needed
                self._evict_if_needed(entry.size)

                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    # Serialize entry using safe JSON
                    entry_dict = entry.to_dict()
                    entry_data = json.dumps(entry_dict).encode('utf-8')

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries
                        (key, value, size, created_at, accessed_at, 
                         access_count, ttl, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            entry_data,
                            entry.size,
                            entry.created_at.timestamp(),
                            entry.accessed_at.timestamp(),
                            entry.access_count,
                            entry.ttl,
                            json.dumps(entry.metadata),
                        ),
                    )

                    conn.commit()
                    return True

            except Exception as e:
                logger.error(f"Failed to put in disk cache: {e}")
                return False

    def exists(self, key: str) -> bool:
        """Check if key exists in disk cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE key = ?", (key,))
                    count = cursor.fetchone()[0]
                    return count > 0
            except Exception as e:
                logger.error(f"Failed to check disk cache exists: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete entry from disk.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0

            except Exception as e:
                logger.error(f"Failed to delete from disk cache: {e}")
                return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM cache_entries")
                    count = cursor.fetchone()[0]
                    cursor.execute("DELETE FROM cache_entries")
                    conn.commit()
                    return count

            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
                return 0

    def size(self) -> int:
        """Get cache size in bytes.

        Returns:
            Size in bytes
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(size) FROM cache_entries")
                result = cursor.fetchone()[0]
                return result or 0

        except Exception as e:
            logger.error(f"Failed to get disk cache size: {e}")
            return 0

    def keys(self) -> Set[str]:
        """Get all cache keys.

        Returns:
            Set of keys
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM cache_entries")
                return {row[0] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get disk cache keys: {e}")
            return set()

    def _evict_if_needed(self, required_size: int):
        """Evict entries if needed to make space.

        Args:
            required_size: Size needed in bytes
        """
        current_size = self.size()

        while current_size + required_size > self.max_size_bytes:
            # Evict oldest accessed entry
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT key, size FROM cache_entries
                    ORDER BY accessed_at ASC
                    LIMIT 1
                """
                )

                row = cursor.fetchone()
                if row:
                    cursor.execute("DELETE FROM cache_entries WHERE key = ?", (row[0],))
                    conn.commit()
                    current_size -= row[1]
                else:
                    break


class RedisCache(CacheBackend):
    """Redis-based cache backend for distributed caching."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 3600,
        prefix: str = "transcription:",
    ):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            ttl: Default TTL in seconds
            prefix: Key prefix
        """
        try:
            import redis

            self.redis = redis.from_url(redis_url)
            self.ttl = ttl
            self.prefix = prefix
            logger.info(f"Connected to Redis cache at {redis_url}")
        except ImportError:
            raise ImportError("redis package required for RedisCache")

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        try:
            full_key = f"{self.prefix}{key}"
            data = self.redis.get(full_key)

            if data:
                # Deserialize entry using safe JSON
                try:
                    entry_dict = json.loads(data.decode('utf-8'))
                    from .transcription_cache import CacheEntry
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    # Update access count
                    self.redis.hincrby(f"{full_key}:meta", "access_count", 1)
                    
                    return entry
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                    logger.error(f"Failed to deserialize Redis cache entry: {e}")
                    # Clean up corrupted entry
                    self.redis.delete(full_key, f"{full_key}:meta")
                    return None

            return None

        except Exception as e:
            logger.error(f"Failed to get from Redis: {e}")
            return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry in Redis.

        Args:
            key: Cache key
            entry: Cache entry

        Returns:
            True if successful
        """
        try:
            full_key = f"{self.prefix}{key}"
            # Serialize entry using safe JSON
            entry_dict = entry.to_dict()
            data = json.dumps(entry_dict).encode('utf-8')

            # Set with TTL
            ttl = entry.ttl or self.ttl
            self.redis.setex(full_key, ttl, data)

            # Store metadata separately
            self.redis.hset(
                f"{full_key}:meta",
                mapping={
                    "size": entry.size,
                    "created_at": entry.created_at.timestamp(),
                    "access_count": 0,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to put in Redis: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        try:
            full_key = f"{self.prefix}{key}"
            return self.redis.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Failed to check Redis cache exists: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from Redis.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            full_key = f"{self.prefix}{key}"
            result = self.redis.delete(full_key, f"{full_key}:meta")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete from Redis: {e}")
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        try:
            pattern = f"{self.prefix}*"
            keys = list(self.redis.scan_iter(match=pattern))

            if keys:
                self.redis.delete(*keys)

            return len(keys) // 2  # Divide by 2 for data + meta keys

        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return 0

    def size(self) -> int:
        """Get cache size in bytes.

        Returns:
            Approximate size in bytes
        """
        try:
            # Get all metadata keys
            pattern = f"{self.prefix}*:meta"
            total_size = 0

            for key in self.redis.scan_iter(match=pattern):
                size = self.redis.hget(key, "size")
                if size:
                    total_size += int(size)

            return total_size

        except Exception as e:
            logger.error(f"Failed to get Redis cache size: {e}")
            return 0

    def keys(self) -> Set[str]:
        """Get all cache keys.

        Returns:
            Set of keys
        """
        try:
            pattern = f"{self.prefix}*"
            keys = set()

            for key in self.redis.scan_iter(match=pattern):
                if not key.decode().endswith(":meta"):
                    # Remove prefix
                    clean_key = key.decode()[len(self.prefix) :]
                    keys.add(clean_key)

            return keys

        except Exception as e:
            logger.error(f"Failed to get Redis cache keys: {e}")
            return set()


class HybridCache(CacheBackend):
    """Hybrid cache with multiple tiers (L1: Memory, L2: Disk, L3: Redis)."""

    def __init__(
        self,
        l1_cache: Optional[CacheBackend] = None,
        l2_cache: Optional[CacheBackend] = None,
        l3_cache: Optional[CacheBackend] = None,
    ):
        """Initialize hybrid cache.

        Args:
            l1_cache: Level 1 cache (fastest, smallest)
            l2_cache: Level 2 cache (medium)
            l3_cache: Level 3 cache (slowest, largest)
        """
        self.l1 = l1_cache or InMemoryCache(max_size_mb=100)
        self.l2 = l2_cache or DiskCache(max_size_mb=1000)
        self.l3 = l3_cache  # Optional Redis cache

        self.caches = [self.l1, self.l2]
        if self.l3:
            self.caches.append(self.l3)

        logger.info(f"Initialized HybridCache with {len(self.caches)} levels")

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get from first available cache level.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        for i, cache in enumerate(self.caches):
            entry = cache.get(key)
            if entry:
                # Promote to higher levels
                for j in range(i):
                    self.caches[j].put(key, entry)
                return entry

        return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put in all cache levels.

        Args:
            key: Cache key
            entry: Cache entry

        Returns:
            True if stored in at least one level
        """
        success = False

        for cache in self.caches:
            if cache.put(key, entry):
                success = True

        return success

    def exists(self, key: str) -> bool:
        """Check if key exists in any cache level.

        Args:
            key: Cache key

        Returns:
            True if key exists in any level
        """
        for cache in self.caches:
            if cache.exists(key):
                return True
        return False

    def delete(self, key: str) -> bool:
        """Delete from all cache levels.

        Args:
            key: Cache key

        Returns:
            True if deleted from at least one level
        """
        success = False

        for cache in self.caches:
            if cache.delete(key):
                success = True

        return success

    def clear(self) -> int:
        """Clear all cache levels.

        Returns:
            Total entries cleared
        """
        total = 0

        for cache in self.caches:
            total += cache.clear()

        return total

    def size(self) -> int:
        """Get total size across all levels.

        Returns:
            Total size in bytes
        """
        return sum(cache.size() for cache in self.caches)

    def keys(self) -> Set[str]:
        """Get union of keys from all levels.

        Returns:
            Set of all keys
        """
        all_keys = set()

        for cache in self.caches:
            all_keys.update(cache.keys())

        return all_keys
