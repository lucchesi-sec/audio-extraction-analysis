"""
End-to-end test framework for audio-extraction-analysis.

This module provides comprehensive E2E testing capabilities including:
- CLI command integration tests
- Provider factory validation
- Performance and load testing
- Security testing
- Test data management
"""

from .base import (
    E2ETestBase,
    CLITestMixin,
    MockProviderMixin,
    PerformanceTestMixin,
    SecurityTestMixin,
    TestFile,
    TestResult,
)

__all__ = [
    # Base classes
    "E2ETestBase",
    # Test mixins
    "CLITestMixin",
    "MockProviderMixin",
    "PerformanceTestMixin",
    "SecurityTestMixin",
    # Data classes
    "TestFile",
    "TestResult",
]