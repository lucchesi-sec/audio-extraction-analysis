"""Tests for console UI components."""

import io
import json
from unittest.mock import patch

from src.ui.console import ConsoleManager


class TestConsoleManager:
    """Test console manager functionality."""

    def test_json_output_mode(self):
        """Test JSON output mode for stage events."""
        console_manager = ConsoleManager(json_output=True)

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            console_manager.print_stage("Test Stage", "starting")

        output_line = captured.getvalue().strip()
        data = json.loads(output_line)
        assert data["stage"] == "Test Stage"
        assert data["status"] == "starting"

    def test_progress_context_fallback(self):
        """Test progress context in non-TTY environment falls back gracefully."""
        console_manager = ConsoleManager()
        console_manager.is_tty = False

        with console_manager.progress_context("Test Progress") as progress:
            assert hasattr(progress, "update")
            # Should not raise
            progress.update(50, 100)

    @patch("src.ui.console.sys.stderr.isatty")
    def test_tty_detection(self, mock_isatty):
        """Test TTY detection toggles modes."""
        mock_isatty.return_value = True
        cm = ConsoleManager()
        assert cm.is_tty is True

        mock_isatty.return_value = False
        cm = ConsoleManager()
        assert cm.is_tty is False
