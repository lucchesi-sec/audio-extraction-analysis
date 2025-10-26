"""Test suite for src.utils package initialization."""

import pytest


class TestUtilsPackage:
    """Test src.utils package module structure and exports."""

    def test_module_import(self):
        """Test that src.utils module can be imported successfully."""
        import src.utils
        assert src.utils is not None

    def test_module_docstring(self):
        """Test that src.utils module has proper docstring."""
        import src.utils
        assert src.utils.__doc__ is not None
        assert isinstance(src.utils.__doc__, str)
        assert len(src.utils.__doc__.strip()) > 0

    def test_docstring_content(self):
        """Test that docstring describes the module purpose."""
        import src.utils
        docstring_lower = src.utils.__doc__.lower()

        # Verify the docstring mentions utility or utilities
        expected_keywords = ["utility", "utilities", "audio", "extraction", "analysis"]
        assert any(keyword in docstring_lower for keyword in expected_keywords), \
            f"Docstring should describe utility functionality: {src.utils.__doc__}"

    def test_all_attribute_exists(self):
        """Test that __all__ attribute is defined."""
        import src.utils
        assert hasattr(src.utils, "__all__")
        assert isinstance(src.utils.__all__, list)

    def test_all_attribute_content(self):
        """Test that __all__ contains expected exports."""
        import src.utils
        expected_exports = [
            "RetryConfig",
            "RetryExhaustedError",
            "calculate_delay",
            "is_retriable_exception",
            "retry_async",
            "retry_on_network_error",
            "retry_on_network_error_async",
            "retry_sync",
        ]

        assert set(src.utils.__all__) == set(expected_exports), \
            f"__all__ should contain {expected_exports}, got {src.utils.__all__}"

    def test_retry_config_exported(self):
        """Test that RetryConfig is properly exported."""
        import src.utils
        assert hasattr(src.utils, "RetryConfig")

        # Verify it's the correct class
        from src.utils.retry import RetryConfig
        assert src.utils.RetryConfig is RetryConfig

    def test_retry_exhausted_error_exported(self):
        """Test that RetryExhaustedError is properly exported."""
        import src.utils
        assert hasattr(src.utils, "RetryExhaustedError")

        # Verify it's the correct exception class
        from src.utils.retry import RetryExhaustedError
        assert src.utils.RetryExhaustedError is RetryExhaustedError

    def test_calculate_delay_exported(self):
        """Test that calculate_delay function is properly exported."""
        import src.utils
        assert hasattr(src.utils, "calculate_delay")

        # Verify it's the correct function
        from src.utils.retry import calculate_delay
        assert src.utils.calculate_delay is calculate_delay

    def test_is_retriable_exception_exported(self):
        """Test that is_retriable_exception function is properly exported."""
        import src.utils
        assert hasattr(src.utils, "is_retriable_exception")

        # Verify it's the correct function
        from src.utils.retry import is_retriable_exception
        assert src.utils.is_retriable_exception is is_retriable_exception

    def test_retry_async_exported(self):
        """Test that retry_async decorator is properly exported."""
        import src.utils
        assert hasattr(src.utils, "retry_async")

        # Verify it's the correct function
        from src.utils.retry import retry_async
        assert src.utils.retry_async is retry_async

    def test_retry_on_network_error_exported(self):
        """Test that retry_on_network_error decorator is properly exported."""
        import src.utils
        assert hasattr(src.utils, "retry_on_network_error")

        # Verify it's the correct function
        from src.utils.retry import retry_on_network_error
        assert src.utils.retry_on_network_error is retry_on_network_error

    def test_retry_on_network_error_async_exported(self):
        """Test that retry_on_network_error_async decorator is properly exported."""
        import src.utils
        assert hasattr(src.utils, "retry_on_network_error_async")

        # Verify it's the correct function
        from src.utils.retry import retry_on_network_error_async
        assert src.utils.retry_on_network_error_async is retry_on_network_error_async

    def test_retry_sync_exported(self):
        """Test that retry_sync decorator is properly exported."""
        import src.utils
        assert hasattr(src.utils, "retry_sync")

        # Verify it's the correct function
        from src.utils.retry import retry_sync
        assert src.utils.retry_sync is retry_sync

    def test_direct_import_retry_config(self):
        """Test that RetryConfig can be imported directly from src.utils."""
        from src.utils import RetryConfig
        assert RetryConfig is not None
        assert hasattr(RetryConfig, "__name__")
        assert RetryConfig.__name__ == "RetryConfig"

    def test_direct_import_retry_exhausted_error(self):
        """Test that RetryExhaustedError can be imported directly from src.utils."""
        from src.utils import RetryExhaustedError
        assert RetryExhaustedError is not None
        assert hasattr(RetryExhaustedError, "__name__")
        assert RetryExhaustedError.__name__ == "RetryExhaustedError"

    def test_direct_import_all_functions(self):
        """Test that all retry functions can be imported directly from src.utils."""
        from src.utils import (
            calculate_delay,
            is_retriable_exception,
            retry_async,
            retry_on_network_error,
            retry_on_network_error_async,
            retry_sync,
        )

        assert calculate_delay is not None
        assert is_retriable_exception is not None
        assert retry_async is not None
        assert retry_on_network_error is not None
        assert retry_on_network_error_async is not None
        assert retry_sync is not None

    def test_wildcard_import(self):
        """Test that wildcard import works correctly."""
        # Import with wildcard should only import items in __all__
        namespace = {}
        exec("from src.utils import *", namespace)

        # Check that only expected items are imported (plus builtins)
        imported_names = [name for name in namespace.keys() if not name.startswith("__")]
        expected_names = [
            "RetryConfig",
            "RetryExhaustedError",
            "calculate_delay",
            "is_retriable_exception",
            "retry_async",
            "retry_on_network_error",
            "retry_on_network_error_async",
            "retry_sync",
        ]

        assert set(imported_names) == set(expected_names), \
            f"Wildcard import should only import {expected_names}, got {imported_names}"

    def test_no_unexpected_public_exports(self):
        """Test that module doesn't expose unexpected public attributes."""
        import src.utils

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(src.utils) if not attr.startswith('_')]

        # Should have the documented exports plus the retry submodule that was imported from
        expected_attrs = [
            "RetryConfig",
            "RetryExhaustedError",
            "calculate_delay",
            "is_retriable_exception",
            "retry",
            "retry_async",
            "retry_on_network_error",
            "retry_on_network_error_async",
            "retry_sync",
        ]
        assert set(public_attrs) == set(expected_attrs), \
            f"Module should export {expected_attrs}, got {public_attrs}"

    def test_retry_config_is_class(self):
        """Test that RetryConfig is actually a class."""
        from src.utils import RetryConfig
        assert isinstance(RetryConfig, type), "RetryConfig should be a class"

    def test_retry_exhausted_error_is_exception(self):
        """Test that RetryExhaustedError is actually an exception class."""
        from src.utils import RetryExhaustedError
        assert isinstance(RetryExhaustedError, type), "RetryExhaustedError should be a class"
        assert issubclass(RetryExhaustedError, Exception), "RetryExhaustedError should be an Exception subclass"

    def test_functions_are_callable(self):
        """Test that exported functions are actually callable."""
        from src.utils import (
            calculate_delay,
            is_retriable_exception,
            retry_async,
            retry_on_network_error,
            retry_on_network_error_async,
            retry_sync,
        )

        assert callable(calculate_delay), "calculate_delay should be callable"
        assert callable(is_retriable_exception), "is_retriable_exception should be callable"
        assert callable(retry_async), "retry_async should be callable"
        assert callable(retry_on_network_error), "retry_on_network_error should be callable"
        assert callable(retry_on_network_error_async), "retry_on_network_error_async should be callable"
        assert callable(retry_sync), "retry_sync should be callable"

    def test_import_does_not_raise(self):
        """Test that importing the module does not raise any exceptions."""
        try:
            import src.utils
            from src.utils import (
                RetryConfig,
                RetryExhaustedError,
                calculate_delay,
                is_retriable_exception,
                retry_async,
                retry_on_network_error,
                retry_on_network_error_async,
                retry_sync,
            )
        except Exception as e:
            pytest.fail(f"Importing src.utils should not raise exceptions: {e}")
