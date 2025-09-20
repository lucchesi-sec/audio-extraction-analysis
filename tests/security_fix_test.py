#!/usr/bin/env python3
"""
Security test to verify the RCE vulnerability fix.

This test verifies that the cache system now uses safe JSON serialization
instead of unsafe pickle deserialization.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.cache.backends import DiskCache, RedisCache
from src.cache.transcription_cache import CacheKey, CacheEntry
from src.models.transcription import TranscriptionResult


def test_disk_cache_safe_serialization():
    """Test that DiskCache uses safe JSON serialization."""
    # Create a temporary directory for the test cache
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache = DiskCache(cache_dir=cache_dir)
        
        # Create a test transcription result
        transcription = TranscriptionResult(
            transcript="Test transcript",
            duration=10.0,
            generated_at=datetime.now(),
            audio_file="test.mp3",
            provider_name="test_provider"
        )
        
        # Create a cache entry
        cache_key = CacheKey(
            file_hash="test_hash",
            provider="test_provider",
            settings_hash="test_settings"
        )
        
        entry = CacheEntry(
            key=cache_key,
            value=transcription,
            size=1024,
            ttl=3600
        )
        
        # Store in cache
        key_str = str(cache_key)
        success = cache.put(key_str, entry)
        assert success, "Failed to store entry in cache"
        
        # Retrieve from cache
        retrieved_entry = cache.get(key_str)
        assert retrieved_entry is not None, "Failed to retrieve entry from cache"
        assert isinstance(retrieved_entry.value, TranscriptionResult), "Retrieved value is not TranscriptionResult"
        assert retrieved_entry.value.transcript == "Test transcript", "Transcript content mismatch"
        
        print("✓ DiskCache safe serialization test passed")


def test_redis_cache_safe_serialization():
    """Test that RedisCache uses safe JSON serialization (if Redis is available)."""
    try:
        import redis
        
        # Try to connect to Redis (skip if not available)
        try:
            cache = RedisCache()
            # Test connection
            cache.redis.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            print("⚠ Redis not available, skipping RedisCache test")
            return
        
        # Create a test transcription result
        transcription = TranscriptionResult(
            transcript="Test transcript",
            duration=10.0,
            generated_at=datetime.now(),
            audio_file="test.mp3",
            provider_name="test_provider"
        )
        
        # Create a cache entry
        cache_key = CacheKey(
            file_hash="test_hash",
            provider="test_provider", 
            settings_hash="test_settings"
        )
        
        entry = CacheEntry(
            key=cache_key,
            value=transcription,
            size=1024,
            ttl=3600
        )
        
        # Store in cache
        key_str = str(cache_key)
        success = cache.put(key_str, entry)
        assert success, "Failed to store entry in Redis cache"
        
        # Retrieve from cache
        retrieved_entry = cache.get(key_str)
        assert retrieved_entry is not None, "Failed to retrieve entry from Redis cache"
        assert isinstance(retrieved_entry.value, TranscriptionResult), "Retrieved value is not TranscriptionResult"
        assert retrieved_entry.value.transcript == "Test transcript", "Transcript content mismatch"
        
        # Clean up
        cache.delete(key_str)
        
        print("✓ RedisCache safe serialization test passed")
        
    except ImportError:
        print("⚠ Redis package not available, skipping RedisCache test")


def test_no_pickle_imports():
    """Test that pickle is not imported in cache modules."""
    import src.cache.backends as backends_module
    import src.cache.transcription_cache as cache_module
    
    # Check that pickle is not in the module's globals
    assert 'pickle' not in backends_module.__dict__, "pickle is still imported in backends module"
    assert 'pickle' not in cache_module.__dict__, "pickle is still imported in cache module"
    
    print("✓ No pickle imports test passed")


def test_serialization_safety():
    """Test that serialization methods are safe and don't use pickle."""
    # Create test objects
    transcription = TranscriptionResult(
        transcript="Test transcript with special chars: <>\"'&",
        duration=10.5,
        generated_at=datetime.now(),
        audio_file="test file.mp3",
        provider_name="test_provider"
    )
    
    cache_key = CacheKey(
        file_hash="abc123",
        provider="test_provider",
        settings_hash="def456"
    )
    
    entry = CacheEntry(
        key=cache_key,
        value=transcription,
        size=2048,
        ttl=7200
    )
    
    # Test round-trip serialization
    entry_dict = entry.to_dict()
    
    # Verify it's JSON serializable
    json_str = json.dumps(entry_dict)
    loaded_dict = json.loads(json_str)
    
    # Reconstruct from dict
    reconstructed_entry = CacheEntry.from_dict(loaded_dict)
    
    # Verify data integrity
    assert reconstructed_entry.key.file_hash == cache_key.file_hash
    assert reconstructed_entry.key.provider == cache_key.provider
    assert reconstructed_entry.key.settings_hash == cache_key.settings_hash
    assert isinstance(reconstructed_entry.value, TranscriptionResult)
    assert reconstructed_entry.value.transcript == transcription.transcript
    assert reconstructed_entry.value.duration == transcription.duration
    assert reconstructed_entry.size == entry.size
    assert reconstructed_entry.ttl == entry.ttl
    
    print("✓ Serialization safety test passed")


if __name__ == "__main__":
    print("Running security fix verification tests...")
    print("=" * 50)
    
    try:
        test_no_pickle_imports()
        test_serialization_safety() 
        test_disk_cache_safe_serialization()
        test_redis_cache_safe_serialization()
        
        print("=" * 50)
        print("🎉 All security fix tests passed!")
        print("✅ RCE vulnerability has been successfully fixed")
        print("✅ Cache system now uses safe JSON serialization")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        print("⚠ RCE vulnerability fix needs attention")
        raise
