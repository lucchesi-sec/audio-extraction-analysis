"""
Common cache utilities and protocols.

This module contains shared functionality for cache backends including:
- Serialization/deserialization helpers
- TTL and size management utilities
- Key normalization functions
- Cache operation contracts
"""

from __future__ import annotations

import json
import logging
import sys
import zlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

from .transcription_cache import CacheEntry

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseCache(Protocol):
    """Protocol defining the cache backend interface.
    
    All cache backends should implement this protocol to ensure consistency
    and enable contract testing.
    """
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache entry or None if not found or expired
        """
        ...
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry in cache.
        
        Args:
            key: Cache key
            entry: Cache entry to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        ...
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is not expired
        """
        ...
    
    def clear(self) -> int:
        """Clear all entries from cache.
        
        Returns:
            Number of entries cleared
        """
        ...
    
    def size(self) -> int:
        """Get current cache size in bytes.
        
        Returns:
            Size in bytes
        """
        ...
    
    def keys(self) -> Set[str]:
        """Get all cache keys.
        
        Returns:
            Set of all non-expired keys
        """
        ...


class CacheUtils:
    """Utility functions for cache operations."""
    
    @staticmethod
    def normalize_key(key: str) -> str:
        """Normalize cache key for consistent storage.
        
        Args:
            key: Raw cache key
            
        Returns:
            Normalized key string
        """
        # Simple normalization - could be extended with hashing for very long keys
        return key.strip().lower()
    
    @staticmethod
    def calculate_size(value: Any) -> int:
        """Calculate the size of a value in bytes.
        
        Args:
            value: Value to measure
            
        Returns:
            Size estimate in bytes
        """
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        elif isinstance(value, str):
            return len(value.encode('utf-8'))
        
        # For complex objects, use sys.getsizeof as estimate
        return sys.getsizeof(value)
    
    @staticmethod
    def is_expired(entry: CacheEntry) -> bool:
        """Check if a cache entry has expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if expired, False otherwise
        """
        return entry.is_expired()


class SerializationHelper:
    """Helper for safe JSON serialization/deserialization of cache values."""
    
    @staticmethod
    def serialize_entry(entry: CacheEntry, use_compression: bool = False) -> bytes:
        """Serialize cache entry to bytes.
        
        Args:
            entry: Cache entry to serialize
            use_compression: Whether to compress the data
            
        Returns:
            Serialized data as bytes
            
        Raises:
            ValueError: If serialization fails
        """
        try:
            # Convert entry to dict
            entry_dict = entry.to_dict()
            
            # Convert to JSON bytes
            json_bytes = json.dumps(entry_dict, ensure_ascii=False).encode('utf-8')
            
            if use_compression:
                # Compress with zlib
                json_bytes = zlib.compress(json_bytes, level=6)
            
            return json_bytes
            
        except (TypeError, ValueError, zlib.error) as e:
            raise ValueError(f"Failed to serialize cache entry: {e}") from e
    
    @staticmethod
    def deserialize_entry(data: bytes, is_compressed: bool = False) -> Optional[CacheEntry]:
        """Deserialize cache entry from bytes.
        
        Args:
            data: Serialized data
            is_compressed: Whether data is compressed
            
        Returns:
            Cache entry or None if deserialization fails
        """
        try:
            # Decompress if needed
            if is_compressed:
                data = zlib.decompress(data)
            
            # Parse JSON
            entry_dict = json.loads(data.decode('utf-8'))
            
            # Reconstruct entry
            return CacheEntry.from_dict(entry_dict)
            
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, zlib.error, ValueError) as e:
            logger.error(f"Failed to deserialize cache entry: {e}")
            return None
    
    @staticmethod 
    def serialize_value(value: Any, use_compression: bool = False) -> bytes:
        """Serialize a value to bytes safely.
        
        Args:
            value: Value to serialize
            use_compression: Whether to compress
            
        Returns:
            Serialized bytes
        """
        try:
            # Handle different value types
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                # Object with to_dict method
                value_dict = {
                    "type": type(value).__name__,
                    "data": value.to_dict()
                }
            else:
                # Try direct JSON serialization
                try:
                    json.dumps(value)  # Test serializable
                    value_dict = {
                        "type": type(value).__name__, 
                        "data": value
                    }
                except (TypeError, ValueError):
                    # Fallback to string representation
                    value_dict = {
                        "type": "str",
                        "data": str(value)
                    }
            
            json_bytes = json.dumps(value_dict).encode('utf-8')
            
            if use_compression:
                json_bytes = zlib.compress(json_bytes, level=6)
                
            return json_bytes
            
        except Exception as e:
            raise ValueError(f"Failed to serialize value: {e}") from e
    
    @staticmethod
    def deserialize_value(data: bytes, is_compressed: bool = False) -> Any:
        """Deserialize a value from bytes.
        
        Args:
            data: Serialized data
            is_compressed: Whether data is compressed
            
        Returns:
            Original value or None if deserialization fails
        """
        try:
            if is_compressed:
                data = zlib.decompress(data)
                
            value_dict = json.loads(data.decode('utf-8'))
            value_type = value_dict.get("type", "dict")
            value_data = value_dict.get("data")
            
            # Reconstruct based on type
            if value_type == "TranscriptionResult" and isinstance(value_data, dict):
                try:
                    from ..models.transcription import TranscriptionResult
                    return TranscriptionResult.from_dict(value_data)
                except ImportError:
                    # If model not available, return raw dict
                    return value_data
            elif value_type == "str":
                return str(value_data)
            else:
                return value_data
                
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            return None


class TTLManager:
    """Helper for managing TTL (Time To Live) functionality."""
    
    @staticmethod
    def is_expired(entry: CacheEntry, current_time: Optional[float] = None) -> bool:
        """Check if entry has expired based on TTL.
        
        Args:
            entry: Cache entry to check
            current_time: Current timestamp (None for now)
            
        Returns:
            True if expired
        """
        if entry.ttl is None:
            return False
            
        import time
        now = current_time or time.time()
        created_timestamp = entry.created_at.timestamp()
        
        return (now - created_timestamp) > entry.ttl
    
    @staticmethod
    def time_until_expiry(entry: CacheEntry) -> Optional[float]:
        """Get time until entry expires.
        
        Args:
            entry: Cache entry
            
        Returns:
            Seconds until expiry, None if no TTL set
        """
        if entry.ttl is None:
            return None
            
        import time
        now = time.time()
        created_timestamp = entry.created_at.timestamp()
        elapsed = now - created_timestamp
        
        return max(0, entry.ttl - elapsed)


class SizeLimitManager:
    """Helper for managing cache size limits."""
    
    def __init__(self, max_size_bytes: int):
        """Initialize size limit manager.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self._current_size = 0
    
    def can_fit(self, entry_size: int) -> bool:
        """Check if an entry of given size can fit.
        
        Args:
            entry_size: Size of entry in bytes
            
        Returns:
            True if it fits within limits
        """
        return entry_size <= self.max_size_bytes
    
    def would_exceed_limit(self, entry_size: int) -> bool:
        """Check if adding entry would exceed size limit.
        
        Args:
            entry_size: Size of entry to add
            
        Returns:
            True if would exceed limit
        """
        return (self._current_size + entry_size) > self.max_size_bytes
    
    def space_needed_for(self, entry_size: int) -> int:
        """Calculate how much space needs to be freed.
        
        Args:
            entry_size: Size of entry to add
            
        Returns:
            Bytes that need to be freed (0 if fits)
        """
        if not self.would_exceed_limit(entry_size):
            return 0
            
        return (self._current_size + entry_size) - self.max_size_bytes
    
    def add_entry(self, entry_size: int) -> None:
        """Add entry size to current total.
        
        Args:
            entry_size: Size of added entry
        """
        self._current_size += entry_size
    
    def remove_entry(self, entry_size: int) -> None:
        """Remove entry size from current total.
        
        Args:
            entry_size: Size of removed entry
        """
        self._current_size = max(0, self._current_size - entry_size)
    
    def reset(self) -> None:
        """Reset size counter to zero."""
        self._current_size = 0
    
    @property
    def current_size(self) -> int:
        """Get current cache size in bytes."""
        return self._current_size
    
    @property 
    def available_space(self) -> int:
        """Get available space in bytes."""
        return max(0, self.max_size_bytes - self._current_size)
    
    @property
    def utilization_percent(self) -> float:
        """Get cache utilization as percentage."""
        if self.max_size_bytes == 0:
            return 0.0
        return (self._current_size / self.max_size_bytes) * 100
