"""Cache backend implementations."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock, RLock, local
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

        # Thread-local storage for connections (SQLite connections aren't thread-safe)
        self._local = local()

        self._init_database()
        logger.info(f"Initialized DiskCache at {self.cache_dir} with max_size={max_size_mb}MB (WAL mode)")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local database connection.

        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))
            # Enable WAL mode for concurrent reads
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            # Additional optimizations
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
        return self._local.conn

    def _init_database(self):
        """Initialize SQLite database with WAL mode."""
        conn = self._get_connection()
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
                row = self._query_entry(key)
                if row:
                    self._update_access_stats(key)
                    return self._deserialize_entry(row, key)
                return None

            except Exception as e:
                logger.error(f"Failed to get from disk cache: {e}")
                return None

    def _query_entry(self, key: str) -> Optional[tuple]:
        """Query cache entry from database.

        Args:
            key: Cache key

        Returns:
            Database row or None
        """
        conn = self._get_connection()
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

        return cursor.fetchone()

    def _update_access_stats(self, key: str):
        """Update access time and count for an entry.

        Args:
            key: Cache key
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE cache_entries
            SET accessed_at = ?, access_count = access_count + 1
            WHERE key = ?
        """,
            (time.time(), key),
        )
        conn.commit()

    def _deserialize_entry(self, row: tuple, key: str) -> Optional[CacheEntry]:
        """Deserialize cache entry from database row.

        Args:
            row: Database row containing serialized entry
            key: Cache key (used for cleanup on error)

        Returns:
            Deserialized cache entry or None
        """
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
            self._delete_corrupted_entry(key)
            return None

    def _delete_corrupted_entry(self, key: str):
        """Delete corrupted cache entry.

        Args:
            key: Cache key
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        conn.commit()

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

                conn = self._get_connection()
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
                conn = self._get_connection()
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
                conn = self._get_connection()
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
                conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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

    def close(self):
        """Close database connections.

        Should be called when the cache is no longer needed to ensure
        proper cleanup of database connections.
        """
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.close()
                delattr(self._local, 'conn')
            except Exception as e:
                logger.error(f"Failed to close database connection: {e}")
