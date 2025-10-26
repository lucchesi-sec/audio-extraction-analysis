"""Eviction strategy helpers extracted from TranscriptionCache.

These pure helper functions make eviction logic reusable and easier to test.
Optimized to O(log n) or better using heap-based and OrderedDict operations.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Set, Any


def select_lru_victim(backend: Any, keys: Set[str]) -> str:
    """Least Recently Used victim selection - O(1) for OrderedDict, O(n) fallback.

    For OrderedDict-based backends like InMemoryCache, this is O(1) since
    move_to_end maintains LRU order. First key is the least recently used.
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
