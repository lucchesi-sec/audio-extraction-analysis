"""Tests for cache benchmark script.

This module tests the functionality of benchmarks/cache_benchmark.py to ensure
benchmark utilities work correctly and benchmarks can execute without errors.
"""

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

from cache_benchmark import create_sample_entry, benchmark_operation, run_benchmarks
from src.cache.backends import DiskCache
from src.cache.transcription_cache import CacheEntry, CacheKey


class TestCreateSampleEntry:
    """Test the create_sample_entry helper function."""

    def test_creates_valid_cache_entry(self):
        """Test that create_sample_entry creates a valid CacheEntry."""
        entry = create_sample_entry(42)

        assert isinstance(entry, CacheEntry)
        assert isinstance(entry.key, CacheKey)
        assert entry.key.file_hash == "test_hash_42"
        assert entry.key.provider == "benchmark_provider"
        assert entry.key.settings_hash == "settings_42"
        assert entry.value["index"] == 42
        assert entry.value["benchmark"] == "data"
        assert "Sample text 42" in entry.value["transcription"]
        assert entry.size == 500
        assert entry.ttl == 3600
        assert entry.metadata["source"] == "benchmark"
        assert entry.access_count == 0

    def test_creates_unique_entries(self):
        """Test that different indices create different entries."""
        entry1 = create_sample_entry(1)
        entry2 = create_sample_entry(2)

        assert entry1.key.file_hash != entry2.key.file_hash
        assert entry1.key.settings_hash != entry2.key.settings_hash
        assert entry1.value["index"] != entry2.value["index"]

    def test_timestamps_are_recent(self):
        """Test that created entries have recent timestamps."""
        before = datetime.now()
        entry = create_sample_entry(0)
        after = datetime.now()

        assert before <= entry.created_at <= after
        assert before <= entry.accessed_at <= after

    def test_creates_entries_for_zero_index(self):
        """Test that index 0 is handled correctly."""
        entry = create_sample_entry(0)

        assert entry.key.file_hash == "test_hash_0"
        assert entry.value["index"] == 0

    def test_creates_entries_for_large_index(self):
        """Test that large indices are handled correctly."""
        large_index = 999999
        entry = create_sample_entry(large_index)

        assert entry.key.file_hash == f"test_hash_{large_index}"
        assert entry.value["index"] == large_index


class TestBenchmarkOperation:
    """Test the benchmark_operation timing function."""

    def test_measures_operation_performance(self):
        """Test that benchmark_operation measures ops/sec."""
        counter = [0]

        def simple_op():
            counter[0] += 1

        ops_per_sec = benchmark_operation("test", simple_op, 100)

        assert counter[0] == 100  # Operation called 100 times
        assert ops_per_sec > 0  # Should return positive ops/sec

    def test_handles_single_iteration(self):
        """Test benchmark with single iteration."""
        counter = [0]

        def simple_op():
            counter[0] += 1

        ops_per_sec = benchmark_operation("single", simple_op, 1)

        assert counter[0] == 1
        assert ops_per_sec > 0

    def test_handles_slow_operations(self):
        """Test benchmark with slow operations."""
        def slow_op():
            time.sleep(0.01)  # 10ms sleep

        ops_per_sec = benchmark_operation("slow", slow_op, 5)

        # Should be around 100 ops/sec (10ms per op)
        # Allow wide range due to timing variability
        assert 50 <= ops_per_sec <= 200

    def test_handles_fast_operations(self):
        """Test benchmark with very fast operations."""
        def fast_op():
            pass  # Essentially no-op

        ops_per_sec = benchmark_operation("fast", fast_op, 1000)

        # Fast operations should have high ops/sec
        assert ops_per_sec > 1000

    def test_operation_name_in_output(self, capsys):
        """Test that operation name appears in printed output."""
        def simple_op():
            pass

        benchmark_operation("TestOperation", simple_op, 10)

        captured = capsys.readouterr()
        assert "TestOperation" in captured.out

    def test_iteration_count_in_output(self, capsys):
        """Test that iteration count appears in output."""
        def simple_op():
            pass

        benchmark_operation("test", simple_op, 42)

        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_handles_operation_with_exception(self):
        """Test that exceptions in operations are propagated."""
        def failing_op():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            benchmark_operation("failing", failing_op, 5)


class TestRunBenchmarks:
    """Test the main run_benchmarks function."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide a temporary directory for cache tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup handled by tempfile

    def test_run_benchmarks_executes_without_error(self, capsys):
        """Test that run_benchmarks completes without errors."""
        # This is a smoke test - just verify it runs
        # We'll mock parts to speed it up
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_cache._get_connection.return_value = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_cache._get_connection.return_value.cursor.return_value = mock_cursor
            MockCache.return_value = mock_cache

            # This should complete without raising
            run_benchmarks()

        captured = capsys.readouterr()
        assert "DiskCache Performance Benchmark" in captured.out

    def test_benchmarks_create_test_data(self, capsys):
        """Test that benchmarks create the expected test data."""
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_cache._get_connection.return_value = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_cache._get_connection.return_value.cursor.return_value = mock_cursor
            MockCache.return_value = mock_cache

            run_benchmarks()

        captured = capsys.readouterr()
        assert "Created 1000 test entries" in captured.out

    def test_benchmarks_measure_all_operations(self, capsys):
        """Test that all benchmark operations are measured."""
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_cache._get_connection.return_value = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_cache._get_connection.return_value.cursor.return_value = mock_cursor
            MockCache.return_value = mock_cache

            run_benchmarks()

        captured = capsys.readouterr()

        # Verify all benchmark sections are present
        assert "Sequential Writes" in captured.out
        assert "Sequential Reads" in captured.out
        assert "Mixed Operations" in captured.out
        assert "Concurrent Reads" in captured.out
        assert "exists() operation" in captured.out

    def test_benchmarks_display_summary(self, capsys):
        """Test that benchmark summary is displayed."""
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_cache._get_connection.return_value = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_cache._get_connection.return_value.cursor.return_value = mock_cursor
            MockCache.return_value = mock_cache

            run_benchmarks()

        captured = capsys.readouterr()
        assert "Summary" in captured.out
        assert "Sequential writes:" in captured.out
        assert "ops/sec" in captured.out

    def test_benchmarks_verify_wal_mode(self, capsys):
        """Test that benchmarks verify WAL mode status."""
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_conn.cursor.return_value = mock_cursor
            mock_cache._get_connection.return_value = mock_conn
            MockCache.return_value = mock_cache

            run_benchmarks()

        captured = capsys.readouterr()
        assert "SQLite journal mode:" in captured.out
        assert "WAL mode enabled:" in captured.out

    def test_benchmarks_cleanup_resources(self):
        """Test that benchmarks clean up cache resources."""
        with patch("cache_benchmark.DiskCache") as MockCache:
            mock_cache = MagicMock()
            mock_cache._get_connection.return_value = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("WAL",)
            mock_cache._get_connection.return_value.cursor.return_value = mock_cursor
            MockCache.return_value = mock_cache

            run_benchmarks()

            # Verify close was called
            mock_cache.close.assert_called_once()

    def test_benchmarks_with_actual_cache(self, temp_cache_dir):
        """Integration test with actual DiskCache instance."""
        # Patch the tempfile.mkdtemp to use our fixture
        with patch("tempfile.mkdtemp", return_value=str(temp_cache_dir)):
            # Run a minimal version by patching iteration counts
            with patch("cache_benchmark.benchmark_operation") as mock_bench:
                # Make benchmark_operation fast
                mock_bench.return_value = 1000.0

                # This will use real DiskCache but mocked benchmarks
                run_benchmarks()

                # Verify benchmark_operation was called for each test
                assert mock_bench.call_count >= 4  # At least 4 benchmark types


class TestBenchmarkIntegration:
    """Integration tests for complete benchmark workflow."""

    def test_create_entry_and_benchmark_together(self):
        """Test that created entries work correctly in benchmarks."""
        temp_dir = Path(tempfile.mkdtemp())
        cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

        # Create and store entries
        entries = [create_sample_entry(i) for i in range(10)]
        keys = [f"key_{i}" for i in range(10)]

        # Benchmark write operations
        write_counter = [0]

        def write_op():
            idx = write_counter[0] % len(entries)
            cache.put(keys[idx], entries[idx])
            write_counter[0] += 1

        ops_per_sec = benchmark_operation("integration_write", write_op, 10)

        assert ops_per_sec > 0
        assert write_counter[0] == 10

        # Verify entries were stored
        for key in keys:
            assert cache.exists(key)

        cache.close()

    def test_benchmark_read_after_write(self):
        """Test benchmarking read operations after writes."""
        temp_dir = Path(tempfile.mkdtemp())
        cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

        # Write data
        entries = [create_sample_entry(i) for i in range(5)]
        keys = [f"key_{i}" for i in range(5)]

        for i, key in enumerate(keys):
            cache.put(key, entries[i])

        # Benchmark reads
        read_counter = [0]

        def read_op():
            idx = read_counter[0] % len(keys)
            result = cache.get(keys[idx])
            assert result is not None  # Verify data is retrieved
            read_counter[0] += 1

        ops_per_sec = benchmark_operation("integration_read", read_op, 10)

        assert ops_per_sec > 0
        assert read_counter[0] == 10

        cache.close()

    def test_benchmark_mixed_operations(self):
        """Test benchmarking mixed read/write operations."""
        temp_dir = Path(tempfile.mkdtemp())
        cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)

        entries = [create_sample_entry(i) for i in range(5)]
        keys = [f"key_{i}" for i in range(5)]

        # Pre-populate some data
        for i in range(3):
            cache.put(keys[i], entries[i])

        # Benchmark mixed ops
        mixed_counter = [0]

        def mixed_op():
            idx = mixed_counter[0] % len(keys)
            if mixed_counter[0] % 2 == 0:
                cache.get(keys[idx])  # Read
            else:
                cache.put(keys[idx], entries[idx])  # Write
            mixed_counter[0] += 1

        ops_per_sec = benchmark_operation("integration_mixed", mixed_op, 10)

        assert ops_per_sec > 0
        assert mixed_counter[0] == 10

        cache.close()
