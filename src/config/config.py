"""Backward compatibility shim for config imports.

DEPRECATED: This import path is deprecated as of v1.1.0.
Please update imports to:
    from src.config import Config, get_config

This shim will be removed in v2.0.0.

Legacy import patterns still supported:
- from .config.config import Config
- from ..config.config import Config
"""
from __future__ import annotations

# Import from the parent __init__.py which contains the Config
from . import Config, get_config

# Re-export for backward compatibility
__all__ = ["Config", "get_config"]
