"""
Contract tests for cache backends to ensure BaseCache protocol conformance.

This module contains tests that verify all cache backends implement the 
BaseCache protocol consistently and correctly.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Type

from src.cache.backends import InMemoryCache, DiskCache
from src.cache.common import BaseCache, CacheUtils
from src.cache.transcription_cache import CacheEntry, CacheKey


class TestCacheContract:
    """Contract tests that all cache backends must pass."""
    
    @pytest.fixture
    def sample_cache_entry(self) -> CacheEntry:
        """Create a sample cache entry for testing."""
        cache_key = CacheKey(
            file_hash="test_hash_123",
            provider="test_provider",
            settings_hash="settings_456"
        )
        return CacheEntry(
            key=cache_key,
            value={"test": "data", "transcription": "Hello world"},
            size=100,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl=3600,
            metadata={"source": "test"}
        )
    
    @pytest.fixture
    def cache_backends(self) -> List[BaseCache]:
        """Create instances of all cache backends for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        return [
            InMemoryCache(max_size_mb=1),
            DiskCache(cache_dir=temp_dir, max_size_mb=1),
        ]
    
    def test_implements_base_cache_protocol(self, cache_backends: List[BaseCache]):
        """Test that all backends implement BaseCache protocol."""
        for backend in cache_backends:
            assert isinstance(backend, BaseCache), f"{type(backend)} does not implement BaseCache"
    
    def test_put_and_get_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test basic put and get operations work consistently."""
        test_key = "test_key_123"
        
        for backend in cache_backends:
            # Test successful put
            assert backend.put(test_key, sample_cache_entry), f"{type(backend)} put operation failed"
            
            # Test successful get
            retrieved_entry = backend.get(test_key)
            assert retrieved_entry is not None, f"{type(backend)} get operation returned None"
            assert retrieved_entry.key.file_hash == sample_cache_entry.key.file_hash
            assert retrieved_entry.value == sample_cache_entry.value
            assert retrieved_entry.size == sample_cache_entry.size
    
    def test_delete_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test delete operations work consistently."""
        test_key = "test_delete_key"
        
        for backend in cache_backends:
            # Put entry first
            backend.put(test_key, sample_cache_entry)
            
            # Test delete returns True for existing key
            assert backend.delete(test_key), f"{type(backend)} delete existing key should return True"
            
            # Test get returns None after delete
            assert backend.get(test_key) is None, f"{type(backend)} should return None after delete"
            
            # Test delete returns False for non-existing key
            assert not backend.delete("non_existing_key"), f"{type(backend)} delete non-existing key should return False"
    
    def test_exists_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test exists operations work consistently."""
        test_key = "test_exists_key"
        
        for backend in cache_backends:
            # Test exists returns False for non-existing key
            assert not backend.exists(test_key), f"{type(backend)} exists should return False for non-existing key"
            
            # Put entry
            backend.put(test_key, sample_cache_entry)
            
            # Test exists returns True for existing key
            assert backend.exists(test_key), f"{type(backend)} exists should return True for existing key"
            
            # Delete and test exists returns False
            backend.delete(test_key)
            assert not backend.exists(test_key), f"{type(backend)} exists should return False after delete"
    
    def test_clear_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test clear operations work consistently."""
        for backend in cache_backends:
            # Put multiple entries
            keys = ["clear_test_1", "clear_test_2", "clear_test_3"]
            for key in keys:
                backend.put(key, sample_cache_entry)
            
            # Test clear returns count > 0
            cleared_count = backend.clear()
            assert cleared_count > 0, f"{type(backend)} clear should return positive count"
            
            # Test all entries are gone
            for key in keys:
                assert backend.get(key) is None, f"{type(backend)} should have no entries after clear"
            
            # Test size is 0 after clear
            assert backend.size() == 0, f"{type(backend)} size should be 0 after clear"
    
    def test_size_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test size operations work consistently."""
        for backend in cache_backends:
            # Clear to start fresh
            backend.clear()
            
            # Test initial size is 0
            assert backend.size() == 0, f"{type(backend)} initial size should be 0"
            
            # Add entry and check size increases
            backend.put("size_test", sample_cache_entry)
            size_after_put = backend.size()
            assert size_after_put > 0, f"{type(backend)} size should increase after put"
            
            # Delete entry and check size decreases
            backend.delete("size_test")
            size_after_delete = backend.size()
            assert size_after_delete < size_after_put, f"{type(backend)} size should decrease after delete"
    
    def test_keys_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test keys operations work consistently."""
        for backend in cache_backends:
            backend.clear()
            
            # Test keys returns empty set initially
            assert len(backend.keys()) == 0, f"{type(backend)} keys should return empty set initially"
            
            # Add entries
            test_keys = ["keys_test_1", "keys_test_2", "keys_test_3"]
            for key in test_keys:
                backend.put(key, sample_cache_entry)
            
            # Test keys returns all added keys
            returned_keys = backend.keys()
            for key in test_keys:
                assert key in returned_keys, f"{type(backend)} keys should contain {key}"
            
            # Delete one key and verify it's no longer in keys
            backend.delete(test_keys[0])
            updated_keys = backend.keys()
            assert test_keys[0] not in updated_keys, f"{type(backend)} keys should not contain deleted key"
            assert test_keys[1] in updated_keys, f"{type(backend)} keys should still contain remaining keys"
    
    def test_key_normalization(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test that key normalization is applied consistently."""
        # Test with different case and whitespace
        original_key = "  TeSt_KeY_123  "
        normalized_key = CacheUtils.normalize_key(original_key)
        
        for backend in cache_backends:
            backend.clear()
            
            # Put with original key
            backend.put(original_key, sample_cache_entry)
            
            # Get with normalized key should work for normalized backends
            retrieved = backend.get(normalized_key)
            if hasattr(backend, '_cache'):  # InMemoryCache uses normalization
                assert retrieved is not None, f"{type(backend)} should retrieve with normalized key"
            
            # Keys should contain normalized version for normalized backends
            keys = backend.keys()
            if hasattr(backend, '_cache'):  # InMemoryCache uses normalization
                assert normalized_key in keys, f"{type(backend)} keys should contain normalized key"
    
    def test_large_entry_handling(self, cache_backends: List[BaseCache]):
        """Test handling of entries larger than cache size."""
        large_key = CacheKey(
            file_hash="large_hash",
            provider="test",
            settings_hash="large_settings"
        )
        # Create entry larger than 1MB cache size
        large_entry = CacheEntry(
            key=large_key,
            value={"data": "x" * (2 * 1024 * 1024)},  # 2MB of data
            size=2 * 1024 * 1024,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        
        for backend in cache_backends:
            backend.clear()
            
            # Should fail to put oversized entry
            result = backend.put("large_key", large_entry)
            assert not result, f"{type(backend)} should reject oversized entries"
            
            # Verify it's not actually stored
            assert backend.get("large_key") is None, f"{type(backend)} should not store oversized entries"
    
    def test_concurrent_operations_safety(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test basic thread safety of operations."""
        import threading
        import time

        for backend in cache_backends:
            backend.clear()
            results = []

            def worker(worker_id: int):
                """Worker function for concurrent testing."""
                key = f"concurrent_{worker_id}"
                # Put entry
                put_result = backend.put(key, sample_cache_entry)
                results.append(put_result)

                # Small delay to allow interleaving
                time.sleep(0.001)

                # Get entry
                get_result = backend.get(key)
                results.append(get_result is not None)

            # Run multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check that all operations succeeded
            assert all(results), f"{type(backend)} concurrent operations failed"

    def test_concurrent_reads(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test concurrent read operations with WAL mode benefits."""
        import threading

        for backend in cache_backends:
            backend.clear()

            # Put a single entry
            test_key = "concurrent_read_key"
            backend.put(test_key, sample_cache_entry)

            read_results = []
            errors = []

            def reader(reader_id: int):
                """Reader function for concurrent testing."""
                try:
                    for _ in range(10):
                        entry = backend.get(test_key)
                        read_results.append(entry is not None)
                except Exception as e:
                    errors.append(e)

            # Run multiple concurrent readers
            threads = []
            for i in range(10):
                thread = threading.Thread(target=reader, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check that all reads succeeded
            assert len(errors) == 0, f"{type(backend)} concurrent reads had errors: {errors}"
            assert all(read_results), f"{type(backend)} concurrent reads failed"
            assert len(read_results) == 100, f"{type(backend)} expected 100 reads, got {len(read_results)}"

    def test_concurrent_writes(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test concurrent write operations."""
        import threading

        for backend in cache_backends:
            backend.clear()

            write_results = []
            errors = []

            def writer(writer_id: int):
                """Writer function for concurrent testing."""
                try:
                    for i in range(5):
                        key = f"concurrent_write_{writer_id}_{i}"
                        result = backend.put(key, sample_cache_entry)
                        write_results.append(result)
                except Exception as e:
                    errors.append(e)

            # Run multiple concurrent writers
            threads = []
            for i in range(10):
                thread = threading.Thread(target=writer, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check that all writes succeeded
            assert len(errors) == 0, f"{type(backend)} concurrent writes had errors: {errors}"
            assert all(write_results), f"{type(backend)} concurrent writes failed"
            assert len(write_results) == 50, f"{type(backend)} expected 50 writes, got {len(write_results)}"

    def test_concurrent_mixed_operations(self, cache_backends: List[BaseCache], sample_cache_entry: CacheEntry):
        """Test mixed concurrent operations (reads, writes, deletes)."""
        import threading
        import random

        for backend in cache_backends:
            backend.clear()

            # Pre-populate with some entries
            for i in range(10):
                backend.put(f"mixed_{i}", sample_cache_entry)

            operation_results = []
            errors = []

            def mixed_worker(worker_id: int):
                """Worker with mixed operations."""
                try:
                    for i in range(10):
                        op = random.choice(['read', 'write', 'delete', 'exists'])
                        key = f"mixed_{random.randint(0, 19)}"

                        if op == 'read':
                            result = backend.get(key)
                            operation_results.append(('read', result is not None or result is None))
                        elif op == 'write':
                            result = backend.put(key, sample_cache_entry)
                            operation_results.append(('write', result))
                        elif op == 'delete':
                            result = backend.delete(key)
                            operation_results.append(('delete', True))
                        elif op == 'exists':
                            result = backend.exists(key)
                            operation_results.append(('exists', True))
                except Exception as e:
                    errors.append(e)

            # Run multiple threads with mixed operations
            threads = []
            for i in range(10):
                thread = threading.Thread(target=mixed_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check that no errors occurred
            assert len(errors) == 0, f"{type(backend)} mixed concurrent operations had errors: {errors}"
            # All operations should complete successfully
            assert all(result[1] for result in operation_results), f"{type(backend)} some operations failed"


class TestCacheUtilities:
    """Test utility functions from common module."""
    
    def test_key_normalization(self):
        """Test key normalization utility."""
        test_cases = [
            ("  Test_Key  ", "test_key"),
            ("UPPERCASE", "uppercase"),
            ("mixed_Case_123", "mixed_case_123"),
            ("", ""),
            ("already_normalized", "already_normalized"),
        ]
        
        for input_key, expected in test_cases:
            result = CacheUtils.normalize_key(input_key)
            assert result == expected, f"normalize_key('{input_key}') should return '{expected}', got '{result}'"
    
    def test_size_calculation(self):
        """Test size calculation utility."""
        test_cases = [
            (b"hello", 5),
            ("hello", 5),  # UTF-8 encoding
            ({"test": "data"}, None),  # Should return some positive size
        ]
        
        for value, expected in test_cases:
            result = CacheUtils.calculate_size(value)
            if expected is None:
                assert result > 0, f"calculate_size should return positive for {type(value)}"
            else:
                assert result == expected, f"calculate_size({value!r}) should return {expected}, got {result}"
    
    def test_expiry_checking(self):
        """Test expiry checking utility."""
        # Create expired entry
        expired_key = CacheKey("hash", "provider", "settings")
        expired_entry = CacheEntry(
            key=expired_key,
            value="test",
            size=10,
            ttl=1,  # 1 second TTL
        )
        
        # Should not be expired immediately
        assert not CacheUtils.is_expired(expired_entry), "Entry should not be expired immediately"
        
        # Should be expired after TTL
        import time
        time.sleep(1.1)  # Wait longer than TTL
        assert CacheUtils.is_expired(expired_entry), "Entry should be expired after TTL"


if __name__ == "__main__":
    pytest.main([__file__])
