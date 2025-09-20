#!/usr/bin/env python3
"""Test basic cache functionality after security fix."""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cache.transcription_cache import TranscriptionCache, CacheKey, CacheEntry
from cache.backends import InMemoryCache, DiskCache


def test_cache_functionality():
    """Test basic cache operations."""
    print("Testing basic cache functionality...")
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        test_file = Path(f.name)
    
    try:
        # Test in-memory cache (disable compression to avoid import issues in test)
        cache = TranscriptionCache(backends=[InMemoryCache()], enable_compression=False)
        
        # Test cache key generation
        settings = {"model": "whisper", "language": "en"}
        cache_key = CacheKey.from_file(test_file, "openai", settings)
        print(f"✓ Cache key generated: {cache_key}")
        
        # Test cache put/get
        test_result = {"text": "Hello world", "confidence": 0.95}
        success = cache.put(test_file, "openai", settings, test_result)
        print(f"✓ Cache put success: {success}")
        
        # Test cache get
        cached_result = cache.get(test_file, "openai", settings)
        print(f"✓ Cache get result: {cached_result}")
        
        # Verify result matches
        assert cached_result == test_result, f"Expected {test_result}, got {cached_result}"
        print("✓ Cache result verification passed")
        
        # Test cache stats
        stats = cache.get_stats()
        print(f"✓ Cache stats: hits={stats.hits}, misses={stats.misses}")
        
        # Test disk cache (also disable compression)
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = TranscriptionCache(backends=[DiskCache(temp_dir)], enable_compression=False)
            
            # Test put/get with disk cache
            success = disk_cache.put(test_file, "openai", settings, test_result)
            print(f"✓ Disk cache put success: {success}")
            
            cached_result = disk_cache.get(test_file, "openai", settings)
            print(f"✓ Disk cache get result: {cached_result}")
            
            assert cached_result == test_result, f"Expected {test_result}, got {cached_result}"
            print("✓ Disk cache result verification passed")
        
        print("\n🎉 All cache functionality tests passed!")
        
    finally:
        # Clean up
        test_file.unlink()


if __name__ == "__main__":
    test_cache_functionality()
