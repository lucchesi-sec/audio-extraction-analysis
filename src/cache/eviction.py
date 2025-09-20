"""Eviction strategy helpers extracted from TranscriptionCache.

These pure helper functions make eviction logic reusable and easier to test.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

from typing import Any


def select_lru_victim(backend: Any, keys: Set[str]) -> str:
    """Least Recently Used victim selection."""
    oldest_key: Optional[str] = None
    oldest_time = datetime.now()
    for key in keys:
        entry = backend.get(key)
        if entry and entry.accessed_at < oldest_time:
            oldest_time = entry.accessed_at
            oldest_key = key
    return oldest_key or next(iter(keys))


def select_lfu_victim(backend: Any, keys: Set[str]) -> str:
    """Least Frequently Used victim selection."""
    least_used_key: Optional[str] = None
    least_count = float("inf")
    for key in keys:
        entry = backend.get(key)
        if entry and entry.access_count < least_count:
            least_count = entry.access_count
            least_used_key = key
    return least_used_key or next(iter(keys))


def select_ttl_victim(backend: Any, keys: Set[str]) -> str:
    """TTL-based victim selection (closest to expiry)."""
    soonest_key: Optional[str] = None
    soonest_expiry = float("inf")
    for key in keys:
        entry = backend.get(key)
        if entry and entry.ttl:
            remaining = entry.ttl - entry.age_seconds()
            if remaining < soonest_expiry:
                soonest_expiry = remaining
                soonest_key = key
    return soonest_key or next(iter(keys))


def select_size_victim(backend: Any, keys: Set[str]) -> str:
    """Select largest entry as victim."""
    largest_key: Optional[str] = None
    largest_size = 0
    for key in keys:
        entry = backend.get(key)
        if entry and entry.size > largest_size:
            largest_size = entry.size
            largest_key = key
    return largest_key or next(iter(keys))


def select_fifo_victim(backend: Any, keys: Set[str]) -> str:
    """First-In-First-Out (oldest creation time) victim selection."""
    oldest_key: Optional[str] = None
    oldest_time = datetime.now()
    for key in keys:
        entry = backend.get(key)
        if entry and entry.created_at < oldest_time:
            oldest_time = entry.created_at
            oldest_key = key
    return oldest_key or next(iter(keys))
