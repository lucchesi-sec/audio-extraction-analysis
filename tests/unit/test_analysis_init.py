"""Test suite for src.analysis package initialization."""

import pytest


class TestAnalysisPackage:
    """Test src.analysis package module structure and exports."""

    def test_module_import(self):
        """Test that src.analysis module can be imported successfully."""
        import src.analysis
        assert src.analysis is not None

    def test_module_docstring(self):
        """Test that src.analysis module has proper docstring."""
        import src.analysis
        assert src.analysis.__doc__ is not None
        assert isinstance(src.analysis.__doc__, str)
        assert len(src.analysis.__doc__.strip()) > 0

    def test_docstring_content(self):
        """Test that docstring describes the module purpose."""
        import src.analysis
        docstring_lower = src.analysis.__doc__.lower()

        # Verify the docstring mentions analysis or reports
        expected_keywords = ["analysis", "report", "transcription"]
        assert any(keyword in docstring_lower for keyword in expected_keywords), \
            f"Docstring should describe analysis functionality: {src.analysis.__doc__}"

    def test_all_attribute_exists(self):
        """Test that __all__ attribute is defined."""
        import src.analysis
        assert hasattr(src.analysis, "__all__")
        assert isinstance(src.analysis.__all__, list)

    def test_all_attribute_content(self):
        """Test that __all__ contains expected exports."""
        import src.analysis
        expected_exports = ["ConciseAnalyzer", "FullAnalyzer"]

        assert set(src.analysis.__all__) == set(expected_exports), \
            f"__all__ should contain {expected_exports}, got {src.analysis.__all__}"

    def test_concise_analyzer_exported(self):
        """Test that ConciseAnalyzer is properly exported."""
        import src.analysis
        assert hasattr(src.analysis, "ConciseAnalyzer")

        # Verify it's the correct class
        from src.analysis.concise_analyzer import ConciseAnalyzer
        assert src.analysis.ConciseAnalyzer is ConciseAnalyzer

    def test_full_analyzer_exported(self):
        """Test that FullAnalyzer is properly exported."""
        import src.analysis
        assert hasattr(src.analysis, "FullAnalyzer")

        # Verify it's the correct class
        from src.analysis.full_analyzer import FullAnalyzer
        assert src.analysis.FullAnalyzer is FullAnalyzer

    def test_direct_import_concise_analyzer(self):
        """Test that ConciseAnalyzer can be imported directly from src.analysis."""
        from src.analysis import ConciseAnalyzer
        assert ConciseAnalyzer is not None
        assert hasattr(ConciseAnalyzer, "__name__")
        assert ConciseAnalyzer.__name__ == "ConciseAnalyzer"

    def test_direct_import_full_analyzer(self):
        """Test that FullAnalyzer can be imported directly from src.analysis."""
        from src.analysis import FullAnalyzer
        assert FullAnalyzer is not None
        assert hasattr(FullAnalyzer, "__name__")
        assert FullAnalyzer.__name__ == "FullAnalyzer"

    def test_wildcard_import(self):
        """Test that wildcard import works correctly."""
        # Import with wildcard should only import items in __all__
        namespace = {}
        exec("from src.analysis import *", namespace)

        # Check that only expected items are imported (plus builtins)
        imported_names = [name for name in namespace.keys() if not name.startswith("__")]
        expected_names = ["ConciseAnalyzer", "FullAnalyzer"]

        assert set(imported_names) == set(expected_names), \
            f"Wildcard import should only import {expected_names}, got {imported_names}"

    def test_no_unexpected_public_exports(self):
        """Test that module doesn't expose unexpected public attributes."""
        import src.analysis

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(src.analysis) if not attr.startswith('_')]

        # Should have the documented exports plus the submodules that were imported from
        expected_attrs = ["ConciseAnalyzer", "FullAnalyzer", "concise_analyzer", "full_analyzer"]
        assert set(public_attrs) == set(expected_attrs), \
            f"Module should export {expected_attrs}, got {public_attrs}"

    def test_analyzer_classes_are_classes(self):
        """Test that exported analyzers are actually classes."""
        from src.analysis import ConciseAnalyzer, FullAnalyzer

        assert isinstance(ConciseAnalyzer, type), "ConciseAnalyzer should be a class"
        assert isinstance(FullAnalyzer, type), "FullAnalyzer should be a class"

    def test_import_does_not_raise(self):
        """Test that importing the module does not raise any exceptions."""
        try:
            import src.analysis
            from src.analysis import ConciseAnalyzer, FullAnalyzer
        except Exception as e:
            pytest.fail(f"Importing src.analysis should not raise exceptions: {e}")
