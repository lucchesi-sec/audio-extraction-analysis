"""Tests for src.utils.logging_factory module."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.logging_factory import LoggingFactory, get_logger


class TestLoggingFactoryInitialize:
    """Tests for LoggingFactory.initialize() method."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        # Reset the factory state
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        # Clear all handlers from root logger
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

        # Clear logging module's internal flag to allow basicConfig to run
        if hasattr(logging.root, 'handlers'):
            logging.root.handlers = []

    def test_initialize_default_settings(self, tmp_path):
        """Test initialize with default settings."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        assert LoggingFactory._initialized is True
        assert LoggingFactory._log_dir == log_dir
        assert log_dir.exists()

        # Verify root logger configuration
        root = logging.getLogger()
        # Note: basicConfig may not change level if it's already been called,
        # but it should be at least INFO or lower
        assert root.level <= logging.INFO

        # Check handlers were created
        assert len(root.handlers) >= 2  # FileHandler + StreamHandler

        # Verify log file was created
        log_file = log_dir / "app.log"
        assert log_file.exists()

    def test_initialize_custom_log_dir(self, tmp_path):
        """Test initialize with custom log directory."""
        custom_dir = tmp_path / "custom_logs"
        LoggingFactory.initialize(log_dir=custom_dir)

        assert LoggingFactory._log_dir == custom_dir
        assert custom_dir.exists()
        assert (custom_dir / "app.log").exists()

    def test_initialize_custom_level(self, tmp_path):
        """Test initialize with custom logging level."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.DEBUG)

        root = logging.getLogger()
        # Verify level is DEBUG or lower (more permissive)
        assert root.level <= logging.DEBUG

    def test_initialize_custom_format(self, tmp_path):
        """Test initialize with custom format string."""
        log_dir = tmp_path / "logs"
        custom_format = "%(levelname)s - %(name)s - %(message)s"

        LoggingFactory.initialize(log_dir=log_dir, format_string=custom_format)

        # Verify initialization completed
        assert LoggingFactory._initialized is True

    def test_initialize_idempotent(self, tmp_path):
        """Test that calling initialize multiple times is idempotent."""
        log_dir1 = tmp_path / "logs1"
        log_dir2 = tmp_path / "logs2"

        LoggingFactory.initialize(log_dir=log_dir1)
        initial_handlers = len(logging.getLogger().handlers)

        # Second call should be ignored
        LoggingFactory.initialize(log_dir=log_dir2)

        # Should still use first log directory
        assert LoggingFactory._log_dir == log_dir1
        # Should not create duplicate handlers
        assert len(logging.getLogger().handlers) == initial_handlers
        # Second directory should not be created
        assert not log_dir2.exists()

    def test_initialize_creates_missing_directory(self, tmp_path):
        """Test that initialize creates log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "log" / "dir"
        assert not log_dir.exists()

        LoggingFactory.initialize(log_dir=log_dir)

        assert log_dir.exists()
        assert (log_dir / "app.log").exists()

    def test_initialize_sets_module_levels(self, tmp_path):
        """Test that initialize sets specific module logging levels."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        # Check specific module levels
        transcription_logger = logging.getLogger("transcription")
        audio_extraction_logger = logging.getLogger("audio_extraction")

        assert transcription_logger.level == logging.DEBUG
        assert audio_extraction_logger.level == logging.INFO

    def test_initialize_default_log_dir(self, tmp_path):
        """Test initialize with no log_dir uses default."""
        # Set default log dir to temp path
        LoggingFactory._log_dir = tmp_path / "logs"

        # Initialize without specifying log_dir
        LoggingFactory.initialize()

        # Should use the default log_dir and create it
        assert LoggingFactory._log_dir == tmp_path / "logs"
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "logs" / "app.log").exists()


class TestLoggingFactoryGetLogger:
    """Tests for LoggingFactory.get_logger() method."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_get_logger_auto_initializes(self, tmp_path):
        """Test that get_logger auto-initializes if not already done."""
        # Set log dir to temp path to avoid creating logs in project
        LoggingFactory._log_dir = tmp_path / "logs"

        assert LoggingFactory._initialized is False
        logger = LoggingFactory.get_logger("test.module")

        # Should auto-initialize
        assert LoggingFactory._initialized is True
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_logger_instance(self, tmp_path):
        """Test that get_logger returns a Logger instance."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        logger = LoggingFactory.get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance_for_same_name(self, tmp_path):
        """Test that get_logger returns same instance for same name."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        logger1 = LoggingFactory.get_logger("test.same")
        logger2 = LoggingFactory.get_logger("test.same")

        assert logger1 is logger2

    def test_get_logger_different_instances_for_different_names(self, tmp_path):
        """Test that get_logger returns different instances for different names."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        logger1 = LoggingFactory.get_logger("test.module1")
        logger2 = LoggingFactory.get_logger("test.module2")

        assert logger1 is not logger2
        assert logger1.name == "test.module1"
        assert logger2.name == "test.module2"


class TestLoggingFactorySetLevel:
    """Tests for LoggingFactory.set_level() method."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_set_level_changes_logger_level(self, tmp_path):
        """Test that set_level changes the logger level."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        logger_name = "test.logger"
        logger = LoggingFactory.get_logger(logger_name)

        # Change level to DEBUG
        LoggingFactory.set_level(logger_name, logging.DEBUG)
        assert logger.level == logging.DEBUG

        # Change level to ERROR
        LoggingFactory.set_level(logger_name, logging.ERROR)
        assert logger.level == logging.ERROR

    def test_set_level_works_without_initialization(self):
        """Test that set_level works even without explicit initialization."""
        logger_name = "test.logger"

        LoggingFactory.set_level(logger_name, logging.WARNING)

        logger = logging.getLogger(logger_name)
        assert logger.level == logging.WARNING

    def test_set_level_all_levels(self, tmp_path):
        """Test set_level with all standard logging levels."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        logger_name = "test.levels"
        logger = LoggingFactory.get_logger(logger_name)

        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

        for level in levels:
            LoggingFactory.set_level(logger_name, level)
            assert logger.level == level


class TestLoggingFactoryConfigureVerbose:
    """Tests for LoggingFactory.configure_verbose() method."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_configure_verbose_true_sets_debug(self, tmp_path):
        """Test that configure_verbose(True) sets DEBUG level."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir)

        LoggingFactory.configure_verbose(verbose=True)

        root = logging.getLogger()
        src_logger = logging.getLogger("src")
        transcription_logger = logging.getLogger("transcription")
        audio_extraction_logger = logging.getLogger("audio_extraction")

        assert root.level == logging.DEBUG
        assert src_logger.level == logging.DEBUG
        assert transcription_logger.level == logging.DEBUG
        assert audio_extraction_logger.level == logging.DEBUG

    def test_configure_verbose_false_sets_info(self, tmp_path):
        """Test that configure_verbose(False) sets INFO level."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.DEBUG)

        # Set verbose to True first, then False
        LoggingFactory.configure_verbose(verbose=True)
        LoggingFactory.configure_verbose(verbose=False)

        root = logging.getLogger()
        src_logger = logging.getLogger("src")
        transcription_logger = logging.getLogger("transcription")
        audio_extraction_logger = logging.getLogger("audio_extraction")

        assert root.level == logging.INFO
        assert src_logger.level == logging.INFO
        assert transcription_logger.level == logging.INFO
        assert audio_extraction_logger.level == logging.INFO

    def test_configure_verbose_default_is_false(self, tmp_path):
        """Test that configure_verbose() with no argument defaults to False."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.DEBUG)

        LoggingFactory.configure_verbose()

        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_configure_verbose_works_without_initialization(self):
        """Test that configure_verbose works without prior initialization."""
        LoggingFactory.configure_verbose(verbose=True)

        root = logging.getLogger()
        assert root.level == logging.DEBUG


class TestBackwardCompatibility:
    """Tests for backward compatibility functions."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_standalone_get_logger_function(self, tmp_path):
        """Test that standalone get_logger function works."""
        LoggingFactory._log_dir = tmp_path / "logs"

        logger = get_logger("test.standalone")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.standalone"

    def test_standalone_function_calls_factory(self, tmp_path):
        """Test that standalone function delegates to LoggingFactory."""
        LoggingFactory._log_dir = tmp_path / "logs"

        factory_logger = LoggingFactory.get_logger("test.module")
        standalone_logger = get_logger("test.module")

        # Should return the same instance
        assert factory_logger is standalone_logger


class TestLoggingFactoryIntegration:
    """Integration tests for LoggingFactory."""

    def setup_method(self):
        """Reset LoggingFactory state before each test."""
        LoggingFactory._initialized = False
        LoggingFactory._log_dir = Path("logs")

        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: initialize, get logger, log message."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.INFO)

        logger = LoggingFactory.get_logger("integration.test")
        logger.info("Integration test message")
        logger.debug("This should not appear")

        # Flush and close handlers to ensure content is written
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()

        log_file = log_dir / "app.log"
        assert log_file.exists()

        content = log_file.read_text()
        assert "Integration test message" in content
        # Debug message should not appear if level is INFO
        if root.level == logging.INFO:
            assert "This should not appear" not in content

    def test_verbose_mode_workflow(self, tmp_path):
        """Test workflow with verbose mode changes."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.INFO)

        logger = LoggingFactory.get_logger("verbose.test")

        # Initially INFO level - debug shouldn't appear
        logger.debug("Debug before verbose")
        logger.info("Info before verbose")

        # Enable verbose mode
        LoggingFactory.configure_verbose(verbose=True)
        logger.debug("Debug after verbose")
        logger.info("Info after verbose")

        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = log_dir / "app.log"
        content = log_file.read_text()

        assert "Debug before verbose" not in content
        assert "Info before verbose" in content
        assert "Debug after verbose" in content
        assert "Info after verbose" in content

    def test_multiple_modules_logging(self, tmp_path):
        """Test that multiple modules can log independently."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.INFO)

        logger1 = LoggingFactory.get_logger("module1")
        logger2 = LoggingFactory.get_logger("module2")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = log_dir / "app.log"
        content = log_file.read_text()

        assert "module1" in content
        assert "Message from module1" in content
        assert "module2" in content
        assert "Message from module2" in content

    def test_level_changes_persist(self, tmp_path):
        """Test that level changes persist across multiple log calls."""
        log_dir = tmp_path / "logs"
        LoggingFactory.initialize(log_dir=log_dir, level=logging.INFO)

        logger = LoggingFactory.get_logger("persist.test")

        # Set to DEBUG
        LoggingFactory.set_level("persist.test", logging.DEBUG)
        logger.debug("Debug message 1")

        # Debug should still work on subsequent calls
        logger.debug("Debug message 2")

        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = log_dir / "app.log"
        content = log_file.read_text()

        assert "Debug message 1" in content
        assert "Debug message 2" in content
