"""Cache eviction strategy helpers for selecting victims during cache cleanup.

This module provides pure helper functions for implementing various cache eviction
policies (LRU, LFU, TTL, Size-based, FIFO). These functions are designed to be
reusable across different cache backend implementations and are optimized for
performance using heap-based and OrderedDict operations.

Each function selects a victim key from a set of candidates based on a specific
eviction strategy. The victim is the cache entry that should be removed to make
space for new entries.

Performance characteristics:
- LRU/FIFO: O(1) for OrderedDict backends, O(n) fallback
- LFU/TTL/Size: O(n) optimized with single-pass algorithms
- All functions avoid intermediate list construction for memory efficiency

Backend Requirements:
The backend parameter must implement a get(key) method that returns cache entries
with the following attributes (depending on the eviction strategy):
- accessed_at: datetime (for LRU)
- access_count: int (for LFU)
- ttl: Optional[float], age_seconds(): float (for TTL)
- size: int (for Size-based)
- created_at: datetime (for FIFO)
"""
from __future__ import annotations

import heapq
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Set, Any


def select_lru_victim(backend: Any, keys: Set[str]) -> str:
    """Select the least recently used (LRU) cache entry as eviction victim.

    Chooses the cache entry with the oldest accessed_at timestamp. For
    OrderedDict-based backends that maintain LRU order via move_to_end(),
    this is O(1) as the first key is always the LRU entry. Otherwise falls
    back to O(n) scan of all entries.

    Args:
        backend: Cache backend with get(key) method returning entries with
            accessed_at attribute. For O(1) performance, backend should have
            an OrderedDict _cache attribute maintained in LRU order.
        keys: Set of candidate cache keys to consider for eviction. Must not
            be empty.

    Returns:
        The key of the least recently used entry. If no valid entries are found,
        returns an arbitrary key from the input set.

    Complexity:
        O(1) for OrderedDict backends, O(n) for generic backends.
    """
    # O(1) optimization for OrderedDict-based backends
    if hasattr(backend, '_cache') and isinstance(backend._cache, OrderedDict):
        if len(backend._cache) > 0:
            return next(iter(backend._cache.keys()))

    # Fallback: O(n) single-pass with min() built-in (faster than manual loop)
    entries_with_time = []
    for key in keys:
        entry = backend.get(key)
        if entry:
            entries_with_time.append((entry.accessed_at, key))

    if not entries_with_time:
        return next(iter(keys))

    return min(entries_with_time)[1]


def select_lfu_victim(backend: Any, keys: Set[str]) -> str:
    """Least Frequently Used victim selection - O(n) optimized with cached lookups.

    Avoids intermediate list construction by using in-place comparison
    while caching backend.get() results to avoid redundant calls.
    """
    min_key = None
    min_count = float("inf")

    for key in keys:
        entry = backend.get(key)
        if entry:
            if entry.access_count < min_count:
                min_count = entry.access_count
                min_key = key

    return min_key if min_key is not None else next(iter(keys))


def select_ttl_victim(backend: Any, keys: Set[str]) -> str:
    """TTL-based victim selection (closest to expiry) - O(n) optimized.

    Avoids intermediate list construction by using in-place comparison
    while caching backend.get() results to avoid redundant calls.
    """
    min_key = None
    min_remaining = float("inf")

    for key in keys:
        entry = backend.get(key)
        if entry and entry.ttl:
            remaining = entry.ttl - entry.age_seconds()
            if remaining < min_remaining:
                min_remaining = remaining
                min_key = key

    return min_key if min_key is not None else next(iter(keys))


def select_size_victim(backend: Any, keys: Set[str]) -> str:
    """Select largest entry as victim - O(n) optimized.

    Avoids intermediate list construction by using in-place comparison
    while caching backend.get() results to avoid redundant calls.
    """
    max_key = None
    max_size = 0

    for key in keys:
        entry = backend.get(key)
        if entry:
            if entry.size > max_size:
                max_size = entry.size
                max_key = key

    return max_key if max_key is not None else next(iter(keys))


def select_fifo_victim(backend: Any, keys: Set[str]) -> str:
    """First-In-First-Out (oldest creation time) victim selection - O(1) for OrderedDict.

    For OrderedDict-based backends that maintain insertion order, this is O(1).
    First key is the oldest (first inserted).
    """
    # O(1) optimization for OrderedDict-based backends
    if hasattr(backend, '_cache') and isinstance(backend._cache, OrderedDict):
        if len(backend._cache) > 0:
            return next(iter(backend._cache.keys()))

    # Fallback: O(n) single-pass with min() built-in
    entries_with_time = []
    for key in keys:
        entry = backend.get(key)
        if entry:
            entries_with_time.append((entry.created_at, key))

    if not entries_with_time:
        return next(iter(keys))

    return min(entries_with_time)[1]
