#!/usr/bin/env python3
"""
Simple test to verify DiskCache works correctly with WAL mode.
"""

import tempfile
from pathlib import Path
from datetime import datetime
import threading
import time

from src.cache.backends import DiskCache
from src.cache.transcription_cache import CacheEntry, CacheKey


def test_basic_operations():
    """Test basic DiskCache operations."""
    print("Testing basic operations...")

    temp_dir = Path(tempfile.mkdtemp())
    cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

    # Create test entry
    key = CacheKey(
        file_hash="test_hash",
        provider="test_provider",
        settings_hash="test_settings"
    )
    entry = CacheEntry(
        key=key,
        value={"test": "data"},
        size=100,
        created_at=datetime.now(),
        accessed_at=datetime.now(),
    )

    # Test put
    assert cache.put("test_key", entry), "Put failed"
    print("  ✓ Put operation successful")

    # Test get
    retrieved = cache.get("test_key")
    assert retrieved is not None, "Get failed"
    assert retrieved.key.file_hash == "test_hash", "Retrieved wrong data"
    print("  ✓ Get operation successful")

    # Test exists
    assert cache.exists("test_key"), "Exists failed"
    print("  ✓ Exists operation successful")

    # Test delete
    assert cache.delete("test_key"), "Delete failed"
    assert not cache.exists("test_key"), "Key still exists after delete"
    print("  ✓ Delete operation successful")

    # Verify WAL mode
    conn = cache._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.upper() == "WAL", f"WAL mode not enabled: {journal_mode}"
    print(f"  ✓ WAL mode enabled: {journal_mode}")

    cache.close()
    print("Basic operations test: PASSED\n")


def test_concurrent_access():
    """Test concurrent access with multiple threads."""
    print("Testing concurrent access...")

    temp_dir = Path(tempfile.mkdtemp())
    cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

    # Create test entries
    for i in range(10):
        key = CacheKey(
            file_hash=f"hash_{i}",
            provider="test",
            settings_hash=f"settings_{i}"
        )
        entry = CacheEntry(
            key=key,
            value={"index": i},
            size=100,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        cache.put(f"key_{i}", entry)

    results = []
    errors = []

    def worker(worker_id: int):
        """Worker thread for concurrent testing."""
        try:
            # Perform multiple operations
            for i in range(50):
                key = f"key_{i % 10}"
                # Read
                entry = cache.get(key)
                results.append(entry is not None)
                # Check exists
                exists = cache.exists(key)
                results.append(exists)
        except Exception as e:
            errors.append(e)

    # Run concurrent workers
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert len(errors) == 0, f"Concurrent access had errors: {errors}"
    assert all(results), "Some concurrent operations failed"
    print(f"  ✓ {len(threads)} threads completed {len(results)} operations successfully")

    cache.close()
    print("Concurrent access test: PASSED\n")


def test_persistent_connection():
    """Test that connections are reused across operations."""
    print("Testing persistent connection...")

    temp_dir = Path(tempfile.mkdtemp())
    cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

    # Get connection multiple times in same thread
    conn1 = cache._get_connection()
    conn2 = cache._get_connection()

    # Should be the same connection object
    assert conn1 is conn2, "Connection not reused in same thread"
    print("  ✓ Connection reused within same thread")

    # Test that different threads get different connections
    other_thread_conn = [None]

    def get_connection_in_thread():
        other_thread_conn[0] = cache._get_connection()

    thread = threading.Thread(target=get_connection_in_thread)
    thread.start()
    thread.join()

    assert other_thread_conn[0] is not None, "Failed to get connection in thread"
    assert other_thread_conn[0] is not conn1, "Connections shared across threads (unsafe!)"
    print("  ✓ Different threads get separate connections")

    cache.close()
    print("Persistent connection test: PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DiskCache WAL Mode Tests")
    print("=" * 60)
    print()

    test_basic_operations()
    test_concurrent_access()
    test_persistent_connection()

    print("=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)
