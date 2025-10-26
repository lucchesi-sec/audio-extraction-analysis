"""Backward compatibility shim for config imports.

This file exists to support existing imports like:
- from .config.config import Config
- from ..config.config import Config
"""
from __future__ import annotations

# Import from the parent __init__.py which contains the Config
from . import Config, get_config

# Re-export for backward compatibility
__all__ = ["Config", "get_config"]
