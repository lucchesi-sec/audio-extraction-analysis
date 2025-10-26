"""Tests for path sanitization utilities."""
import re
from pathlib import Path

import pytest

from src.utils.sanitization import (
    PathSanitizer,
    sanitize_dirname,
    sanitize_filename,
    sanitize_path,
)


class TestPathSanitizerSubprocess:
    """Tests for sanitize_for_subprocess method."""

    def test_sanitize_simple_path(self, tmp_path):
        """Test sanitization of a simple path."""
        test_path = tmp_path / "simple.txt"
        result = PathSanitizer.sanitize_for_subprocess(test_path)

        # Should be quoted and absolute
        assert result.startswith("'") or result.startswith('"') or "/" in result
        assert str(test_path.resolve()) in result.replace("'", "").replace('"', "")

    def test_sanitize_path_with_spaces(self, tmp_path):
        """Test sanitization of path with spaces."""
        test_path = tmp_path / "file with spaces.txt"
        result = PathSanitizer.sanitize_for_subprocess(test_path)

        # Should be properly quoted
        assert "'" in result or '"' in result

    def test_sanitize_path_with_special_chars(self, tmp_path):
        """Test sanitization of path with shell special characters."""
        test_path = tmp_path / "file$with&special|chars.txt"
        result = PathSanitizer.sanitize_for_subprocess(test_path)

        # Should be quoted to protect special chars
        assert "'" in result or '"' in result

    def test_sanitize_string_path(self, tmp_path):
        """Test sanitization of string path."""
        test_path = str(tmp_path / "test.txt")
        result = PathSanitizer.sanitize_for_subprocess(test_path)

        # Should handle string paths
        assert isinstance(result, str)
        assert test_path in result.replace("'", "").replace('"', "")

    def test_sanitize_relative_path(self):
        """Test that relative paths are converted to absolute."""
        result = PathSanitizer.sanitize_for_subprocess("./relative/path.txt")

        # Should be absolute
        resolved = str(Path("./relative/path.txt").resolve())
        assert resolved in result.replace("'", "").replace('"', "")

    def test_sanitize_empty_string_path(self):
        """Test sanitization of empty string path."""
        result = PathSanitizer.sanitize_for_subprocess("")

        # Should handle empty string gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sanitize_path_with_quotes(self):
        """Test path with single and double quotes."""
        result = PathSanitizer.sanitize_for_subprocess("/path/with'quote.txt")

        # Should be properly escaped by shlex.quote
        assert isinstance(result, str)
        # shlex.quote should handle quotes safely

    def test_sanitize_path_with_newline(self, tmp_path):
        """Test that paths with newlines are sanitized."""
        # Pathlib will reject this, but test string path
        result = PathSanitizer.sanitize_for_subprocess("/path/without\nnewline")

        # Should be escaped or handled safely
        assert isinstance(result, str)


class TestPathSanitizerFilename:
    """Tests for sanitize_filename method."""

    def test_sanitize_simple_filename(self):
        """Test sanitization of simple filename."""
        result = PathSanitizer.sanitize_filename("simple.txt")
        assert result == "simple.txt"

    def test_sanitize_filename_with_spaces(self):
        """Test filename with spaces is preserved."""
        result = PathSanitizer.sanitize_filename("file with spaces.txt")
        assert result == "file with spaces.txt"

    def test_sanitize_filename_with_invalid_chars(self):
        """Test removal of invalid characters."""
        result = PathSanitizer.sanitize_filename("file:name*with?invalid<chars>|.txt")

        # Invalid chars should be replaced
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert ".txt" in result  # Extension preserved

    def test_sanitize_filename_custom_replacement(self):
        """Test custom replacement character."""
        result = PathSanitizer.sanitize_filename("file:name*.txt", replacement="-")

        assert ":" not in result
        assert "*" not in result
        assert "-" in result

    def test_sanitize_filename_multiple_replacements_collapsed(self):
        """Test that consecutive replacements are collapsed."""
        result = PathSanitizer.sanitize_filename("file:::name***.txt")

        # Multiple consecutive underscores should be collapsed to one
        assert "___" not in result
        assert "__" not in result

    def test_sanitize_filename_strips_leading_trailing(self):
        """Test stripping of leading/trailing whitespace and replacement chars."""
        result = PathSanitizer.sanitize_filename("  _filename_.txt_  ")

        # Should not start or end with spaces or underscores
        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert not result.startswith("_")

    def test_sanitize_empty_filename(self):
        """Test that empty filename gets default name."""
        result = PathSanitizer.sanitize_filename("")
        assert result == "unnamed"

    def test_sanitize_only_invalid_chars(self):
        """Test filename with only invalid characters."""
        result = PathSanitizer.sanitize_filename("***:::|||")
        assert result == "unnamed"

    def test_sanitize_long_filename(self):
        """Test that long filenames are truncated."""
        long_name = "a" * 300 + ".txt"
        result = PathSanitizer.sanitize_filename(long_name)

        # Should be under max length
        assert len(result) <= 205  # 200 + extension
        assert result.endswith(".txt")  # Extension preserved

    def test_sanitize_long_filename_without_extension(self):
        """Test truncation of long filename without extension."""
        long_name = "a" * 300
        result = PathSanitizer.sanitize_filename(long_name)

        # Should be truncated to max length
        assert len(result) == 200

    def test_sanitize_preserves_dots_and_hyphens(self):
        """Test that dots and hyphens are preserved."""
        result = PathSanitizer.sanitize_filename("file-name.with.dots.txt")
        assert result == "file-name.with.dots.txt"

    def test_sanitize_unicode_filename(self):
        """Test handling of unicode characters."""
        result = PathSanitizer.sanitize_filename("файл.txt")

        # Should handle unicode
        assert len(result) > 0
        assert result != "unnamed"


class TestPathSanitizerDirname:
    """Tests for sanitize_dirname method."""

    def test_sanitize_simple_dirname(self):
        """Test sanitization of simple directory name."""
        result = PathSanitizer.sanitize_dirname("simple")
        assert result == "simple"

    def test_sanitize_dirname_with_spaces(self):
        """Test dirname with spaces is preserved."""
        result = PathSanitizer.sanitize_dirname("dir with spaces")
        assert result == "dir with spaces"

    def test_sanitize_dirname_no_dots(self):
        """Test that dots are removed from directory names."""
        result = PathSanitizer.sanitize_dirname("dir.with.dots")

        # Dots should be replaced (more restrictive than filename)
        assert "." not in result

    def test_sanitize_dirname_invalid_chars(self):
        """Test removal of invalid characters from dirname."""
        result = PathSanitizer.sanitize_dirname("dir:name/with\\invalid*chars")

        # Invalid chars should be replaced
        assert ":" not in result
        assert "/" not in result
        assert "\\" not in result
        assert "*" not in result

    def test_sanitize_dirname_custom_replacement(self):
        """Test custom replacement character."""
        result = PathSanitizer.sanitize_dirname("dir:name", replacement="-")

        assert ":" not in result
        assert "-" in result

    def test_sanitize_dirname_strips_leading_trailing(self):
        """Test stripping of leading/trailing characters."""
        result = PathSanitizer.sanitize_dirname("  _dirname_  ")

        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert not result.startswith("_")

    def test_sanitize_empty_dirname(self):
        """Test that empty dirname gets default name."""
        result = PathSanitizer.sanitize_dirname("")
        assert result == "unnamed_dir"

    def test_sanitize_long_dirname(self):
        """Test that long dirnames are truncated."""
        long_name = "a" * 150
        result = PathSanitizer.sanitize_dirname(long_name)

        # Should be under max length
        assert len(result) <= 100

    def test_sanitize_dirname_only_invalid_chars(self):
        """Test dirname with only invalid characters."""
        result = PathSanitizer.sanitize_dirname("***:::**")
        assert result == "unnamed_dir"


class TestEnsureSafeSubpath:
    """Tests for ensure_safe_subpath method."""

    def test_ensure_safe_subpath_simple(self, tmp_path):
        """Test ensuring safe subpath with simple relative path."""
        base = tmp_path / "base"
        base.mkdir()

        result = PathSanitizer.ensure_safe_subpath(base, "subdir/file.txt")

        # Should be within base
        assert str(result).startswith(str(base.resolve()))
        assert result == base / "subdir" / "file.txt"

    def test_ensure_safe_subpath_prevents_traversal(self, tmp_path):
        """Test that path traversal attempts with .. are sanitized."""
        base = tmp_path / "base"
        base.mkdir()

        # The function removes .. components, so this becomes "outside/file.txt"
        # which is valid within base
        result = PathSanitizer.ensure_safe_subpath(base, "../outside/file.txt")

        # Should still be within base (.. is stripped)
        assert str(result).startswith(str(base.resolve()))
        assert result == base / "outside" / "file.txt"

    def test_ensure_safe_subpath_removes_parent_refs(self, tmp_path):
        """Test that parent directory references are removed."""
        base = tmp_path / "base"
        base.mkdir()

        # Should skip .. components
        result = PathSanitizer.ensure_safe_subpath(base, "dir/../other/file.txt")

        # Should be within base and not escape
        assert str(result).startswith(str(base.resolve()))

    def test_ensure_safe_subpath_removes_current_refs(self, tmp_path):
        """Test that current directory references are removed."""
        base = tmp_path / "base"
        base.mkdir()

        result = PathSanitizer.ensure_safe_subpath(base, "./dir/./file.txt")

        # Should skip . components
        assert result == base / "dir" / "file.txt"

    def test_ensure_safe_subpath_absolute_raises_error(self, tmp_path):
        """Test that absolute paths that escape base raise error."""
        base = tmp_path / "base"
        base.mkdir()

        # Absolute paths that would escape base should raise ValueError
        with pytest.raises(ValueError, match="Path traversal detected"):
            PathSanitizer.ensure_safe_subpath(base, "/some/absolute/path.txt")

    def test_ensure_safe_subpath_complex_traversal_sanitized(self, tmp_path):
        """Test that complex path traversal attempts are sanitized."""
        base = tmp_path / "base"
        base.mkdir()

        # The function removes all .. components, so this becomes "good/bad/file.txt"
        result = PathSanitizer.ensure_safe_subpath(base, "good/../../bad/file.txt")

        # Should still be within base (all .. are stripped)
        assert str(result).startswith(str(base.resolve()))
        assert result == base / "good" / "bad" / "file.txt"

    def test_ensure_safe_subpath_empty_path(self, tmp_path):
        """Test empty subpath returns base directory."""
        base = tmp_path / "base"
        base.mkdir()

        result = PathSanitizer.ensure_safe_subpath(base, "")

        # Should be the base itself
        assert result == base.resolve()


class TestValidatePathSecurity:
    """Tests for validate_path_security method."""

    def test_validate_safe_path(self, tmp_path):
        """Test validation of safe path."""
        safe_path = tmp_path / "safe" / "file.txt"

        # Should not raise
        PathSanitizer.validate_path_security(safe_path)

    def test_validate_path_with_null_byte(self):
        """Test that null bytes are rejected."""
        with pytest.raises(ValueError, match="NUL byte"):
            PathSanitizer.validate_path_security("/path/with\x00null")

    def test_validate_path_with_control_chars(self):
        """Test that control characters are rejected."""
        with pytest.raises(ValueError, match="control characters"):
            PathSanitizer.validate_path_security("/path/with\x01control")

    def test_validate_path_with_semicolon(self):
        """Test that semicolons are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with;semicolon")

    def test_validate_path_with_pipe(self):
        """Test that pipes are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with|pipe")

    def test_validate_path_with_ampersand(self):
        """Test that ampersands are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with&ampersand")

    def test_validate_path_with_backtick(self):
        """Test that backticks are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with`backtick")

    def test_validate_path_with_dollar(self):
        """Test that dollar signs are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with$dollar")

    def test_validate_path_with_redirect(self):
        """Test that redirect operators are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with>redirect")

        with pytest.raises(ValueError, match="Invalid characters"):
            PathSanitizer.validate_path_security("/path/with<redirect")

    def test_validate_path_allows_brackets(self, tmp_path):
        """Test that brackets and parentheses are allowed."""
        safe_path = tmp_path / "file[1].txt"

        # Should not raise (brackets explicitly allowed)
        PathSanitizer.validate_path_security(safe_path)

    def test_validate_path_allows_parentheses(self, tmp_path):
        """Test that parentheses are allowed."""
        safe_path = tmp_path / "file(1).txt"

        # Should not raise
        PathSanitizer.validate_path_security(safe_path)


class TestGetSafeOutputPath:
    """Tests for get_safe_output_path method."""

    def test_get_safe_output_path_default_dir(self, tmp_path):
        """Test getting output path in same directory as input."""
        input_path = tmp_path / "input.mp4"

        result = PathSanitizer.get_safe_output_path(input_path)

        # Should be in same directory
        assert result.parent == tmp_path
        # Should have new suffix
        assert result.suffix == ".output"
        assert result.stem == "input"

    def test_get_safe_output_path_custom_dir(self, tmp_path):
        """Test getting output path in custom directory."""
        input_path = tmp_path / "input.mp4"
        output_dir = tmp_path / "outputs"

        result = PathSanitizer.get_safe_output_path(input_path, output_dir=output_dir)

        # Should be in custom directory
        assert result.parent == output_dir
        # Directory should be created
        assert output_dir.exists()

    def test_get_safe_output_path_custom_suffix(self, tmp_path):
        """Test getting output path with custom suffix."""
        input_path = tmp_path / "input.mp4"

        result = PathSanitizer.get_safe_output_path(input_path, suffix=".wav")

        # Should have custom suffix
        assert result.suffix == ".wav"

    def test_get_safe_output_path_sanitizes_name(self, tmp_path):
        """Test that output filename is sanitized."""
        input_path = tmp_path / "input:with*invalid|chars.mp4"

        result = PathSanitizer.get_safe_output_path(input_path)

        # Name should be sanitized
        assert ":" not in result.name
        assert "*" not in result.name
        assert "|" not in result.name

    def test_get_safe_output_path_creates_nested_dirs(self, tmp_path):
        """Test that nested output directories are created."""
        input_path = tmp_path / "input.mp4"
        output_dir = tmp_path / "level1" / "level2" / "level3"

        result = PathSanitizer.get_safe_output_path(input_path, output_dir=output_dir)

        # All nested directories should be created
        assert output_dir.exists()
        assert result.parent == output_dir


class TestConvenienceFunctions:
    """Tests for backward compatibility convenience functions."""

    def test_sanitize_path_function(self, tmp_path):
        """Test sanitize_path convenience function."""
        test_path = tmp_path / "test.txt"
        result = sanitize_path(test_path)

        # Should call PathSanitizer.sanitize_for_subprocess
        assert isinstance(result, str)
        assert str(test_path.resolve()) in result.replace("'", "").replace('"', "")

    def test_sanitize_filename_function(self):
        """Test sanitize_filename convenience function."""
        result = sanitize_filename("test:file*.txt")

        # Should call PathSanitizer.sanitize_filename
        assert ":" not in result
        assert "*" not in result
        assert ".txt" in result

    def test_sanitize_dirname_function(self):
        """Test sanitize_dirname convenience function."""
        result = sanitize_dirname("test:dir")

        # Should call PathSanitizer.sanitize_dirname
        assert ":" not in result


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_path_with_unicode_and_special_chars(self):
        """Test path with both unicode and special characters."""
        filename = "файл:test*file.txt"
        result = PathSanitizer.sanitize_filename(filename)

        assert ":" not in result
        assert "*" not in result
        assert len(result) > 0

    def test_very_long_path_components(self, tmp_path):
        """Test handling of very long path components."""
        long_filename = "a" * 500 + ".txt"
        input_path = tmp_path / long_filename

        result = PathSanitizer.get_safe_output_path(input_path)

        # Should handle long filenames gracefully
        assert len(result.name) <= 210  # 200 + suffix length

    def test_sanitize_preserves_empty_replacement(self):
        """Test sanitization with empty replacement string."""
        result = PathSanitizer.sanitize_filename("file:name*.txt", replacement="")

        # Should remove invalid chars without replacement
        assert ":" not in result
        assert "*" not in result

    def test_dirname_trailing_replacement_stripped(self):
        """Test that trailing replacement chars are stripped from dirname."""
        result = PathSanitizer.sanitize_dirname("a" * 150, replacement="_")

        # Should be truncated and not end with underscore
        assert len(result) <= 100
        assert not result.endswith("_")
