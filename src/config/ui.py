"""UI and display configuration settings."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .base import BaseConfig, ConfigurationSchema

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Output format options."""

    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"
    MARKDOWN = "markdown"
    HTML = "html"


class ColorScheme(Enum):
    """Color scheme options."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    MONOCHROME = "monochrome"


class ProgressStyle(Enum):
    """Progress bar styles."""

    BAR = "bar"
    SPINNER = "spinner"
    DOTS = "dots"
    PULSE = "pulse"
    NONE = "none"


@dataclass
class UITheme:
    """UI theme configuration."""

    name: str
    primary_color: str
    secondary_color: str
    success_color: str
    warning_color: str
    error_color: str
    info_color: str
    text_color: str
    background_color: str


class UIConfig(BaseConfig):
    """UI and display configuration."""

    # Predefined themes
    _THEMES = {
        "default": UITheme(
            name="default",
            primary_color="#007ACC",
            secondary_color="#40E0D0",
            success_color="#28a745",
            warning_color="#ffc107",
            error_color="#dc3545",
            info_color="#17a2b8",
            text_color="#212529",
            background_color="#ffffff",
        ),
        "dark": UITheme(
            name="dark",
            primary_color="#0d7377",
            secondary_color="#14ffec",
            success_color="#32d74b",
            warning_color="#ffd60a",
            error_color="#ff453a",
            info_color="#0a84ff",
            text_color="#ffffff",
            background_color="#1c1c1e",
        ),
        "high_contrast": UITheme(
            name="high_contrast",
            primary_color="#0000ff",
            secondary_color="#00ff00",
            success_color="#00ff00",
            warning_color="#ffff00",
            error_color="#ff0000",
            info_color="#00ffff",
            text_color="#000000",
            background_color="#ffffff",
        ),
    }

    def __init__(self):
        """Initialize UI configuration."""
        super().__init__()

        # Output settings
        self.output_format = OutputFormat(self.get_value("OUTPUT_FORMAT", "text").lower())
        self.json_indent = int(self.get_value("JSON_INDENT", "2"))
        self.json_sort_keys = self.parse_bool(self.get_value("JSON_SORT_KEYS", "false"))

        # Display settings
        self.verbose = self.parse_bool(self.get_value("VERBOSE", "false"))
        self.quiet = self.parse_bool(self.get_value("QUIET", "false"))
        self.debug = self.parse_bool(self.get_value("DEBUG", "false"))
        self.show_timestamps = self.parse_bool(self.get_value("SHOW_TIMESTAMPS", "true"))
        self.show_progress = self.parse_bool(self.get_value("SHOW_PROGRESS", "true"))

        # Color settings (respects NO_COLOR standard)
        self.no_color = (
            self.parse_bool(self.get_value("NO_COLOR", "false"))
            or os.getenv("NO_COLOR") is not None
        )
        self.force_color = self.parse_bool(self.get_value("FORCE_COLOR", "false"))
        self.color_scheme = ColorScheme(self.get_value("COLOR_SCHEME", "default").lower())

        # Theme
        theme_name = self.get_value("UI_THEME", "default")
        self.theme = self._load_theme(theme_name)

        # Progress bar settings
        self.progress_style = ProgressStyle(self.get_value("PROGRESS_STYLE", "bar").lower())
        self.progress_bar_width = int(self.get_value("PROGRESS_BAR_WIDTH", "40"))
        self.progress_refresh_rate = float(self.get_value("PROGRESS_REFRESH_RATE", "0.1"))
        self.progress_show_eta = self.parse_bool(self.get_value("PROGRESS_SHOW_ETA", "true"))
        self.progress_show_percentage = self.parse_bool(
            self.get_value("PROGRESS_SHOW_PERCENTAGE", "true")
        )

        # Rich output settings
        self.rich_output = self.parse_bool(self.get_value("RICH_OUTPUT", "true"))
        self.rich_tracebacks = self.parse_bool(self.get_value("RICH_TRACEBACKS", "true"))
        self.rich_panel_style = self.get_value("RICH_PANEL_STYLE", "bold")

        # Table display settings
        self.table_style = self.get_value("TABLE_STYLE", "rounded")
        self.table_max_width = int(self.get_value("TABLE_MAX_WIDTH", "120"))
        self.table_show_header = self.parse_bool(self.get_value("TABLE_SHOW_HEADER", "true"))
        self.table_show_footer = self.parse_bool(self.get_value("TABLE_SHOW_FOOTER", "false"))

        # Terminal settings
        self.terminal_width = self._get_terminal_width()
        self.terminal_height = self._get_terminal_height()
        self.clear_screen = self.parse_bool(self.get_value("CLEAR_SCREEN", "false"))

        # Logging display
        self.log_to_console = self.parse_bool(self.get_value("LOG_TO_CONSOLE", "true"))
        self.log_format_style = self.get_value("LOG_FORMAT_STYLE", "default")
        self.log_show_path = self.parse_bool(self.get_value("LOG_SHOW_PATH", "false"))

        # Interactive settings
        self.interactive_mode = self.parse_bool(self.get_value("INTERACTIVE_MODE", "false"))
        self.confirm_actions = self.parse_bool(self.get_value("CONFIRM_ACTIONS", "false"))
        self.auto_complete = self.parse_bool(self.get_value("AUTO_COMPLETE", "true"))

        # Notification settings
        self.desktop_notifications = self.parse_bool(
            self.get_value("DESKTOP_NOTIFICATIONS", "false")
        )
        self.sound_notifications = self.parse_bool(self.get_value("SOUND_NOTIFICATIONS", "false"))

        # Markdown export settings
        self.markdown_include_timestamps = self.parse_bool(
            self.get_value("MARKDOWN_INCLUDE_TIMESTAMPS", "true")
        )
        self.markdown_include_speakers = self.parse_bool(
            self.get_value("MARKDOWN_INCLUDE_SPEAKERS", "true")
        )
        self.markdown_include_confidence = self.parse_bool(
            self.get_value("MARKDOWN_INCLUDE_CONFIDENCE", "false")
        )
        self.markdown_template = self.get_value("MARKDOWN_TEMPLATE", "default")

        # Pager settings
        self.use_pager = self.parse_bool(self.get_value("USE_PAGER", "false"))
        self.pager_command = self.get_value("PAGER", "less -R")

    def _load_theme(self, theme_name: str) -> UITheme:
        """Load UI theme by name.

        Args:
            theme_name: Theme name or "custom"

        Returns:
            UITheme instance
        """
        if theme_name in self._THEMES:
            return self._THEMES[theme_name]

        # Custom theme from environment
        return UITheme(
            name="custom",
            primary_color=self.get_value("THEME_PRIMARY_COLOR", "#007ACC"),
            secondary_color=self.get_value("THEME_SECONDARY_COLOR", "#40E0D0"),
            success_color=self.get_value("THEME_SUCCESS_COLOR", "#28a745"),
            warning_color=self.get_value("THEME_WARNING_COLOR", "#ffc107"),
            error_color=self.get_value("THEME_ERROR_COLOR", "#dc3545"),
            info_color=self.get_value("THEME_INFO_COLOR", "#17a2b8"),
            text_color=self.get_value("THEME_TEXT_COLOR", "#212529"),
            background_color=self.get_value("THEME_BACKGROUND_COLOR", "#ffffff"),
        )

    def _get_terminal_width(self) -> int:
        """Get terminal width.

        Returns:
            Terminal width in characters
        """
        default_width = 80

        # Try environment variable first
        env_width = self.get_value("TERMINAL_WIDTH")
        if env_width:
            try:
                return int(env_width)
            except ValueError:
                pass

        # Try to get from terminal
        try:
            import shutil

            return shutil.get_terminal_size().columns
        except Exception:
            return default_width

    def _get_terminal_height(self) -> int:
        """Get terminal height.

        Returns:
            Terminal height in lines
        """
        default_height = 24

        # Try environment variable first
        env_height = self.get_value("TERMINAL_HEIGHT")
        if env_height:
            try:
                return int(env_height)
            except ValueError:
                pass

        # Try to get from terminal
        try:
            import shutil

            return shutil.get_terminal_size().lines
        except Exception:
            return default_height

    def should_use_color(self) -> bool:
        """Determine if color output should be used.

        Returns:
            True if color should be used
        """
        if self.no_color:
            return False
        if self.force_color:
            return True

        # Check if output is to a terminal
        try:
            import sys

            return sys.stdout.isatty()
        except Exception:
            return False

    def get_color(self, color_type: str) -> str:
        """Get color code for specified type.

        Args:
            color_type: Type of color (primary, success, error, etc.)

        Returns:
            Color code or empty string if colors disabled
        """
        if not self.should_use_color():
            return ""

        color_map = {
            "primary": self.theme.primary_color,
            "secondary": self.theme.secondary_color,
            "success": self.theme.success_color,
            "warning": self.theme.warning_color,
            "error": self.theme.error_color,
            "info": self.theme.info_color,
            "text": self.theme.text_color,
            "background": self.theme.background_color,
        }

        return color_map.get(color_type, "")

    def get_progress_config(self) -> Dict[str, Any]:
        """Get progress bar configuration.

        Returns:
            Progress bar configuration dictionary
        """
        return {
            "style": self.progress_style.value,
            "width": self.progress_bar_width,
            "refresh_rate": self.progress_refresh_rate,
            "show_eta": self.progress_show_eta,
            "show_percentage": self.progress_show_percentage,
            "enabled": self.show_progress and not self.quiet,
        }

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration.

        Returns:
            Output configuration dictionary
        """
        return {
            "format": self.output_format.value,
            "verbose": self.verbose,
            "quiet": self.quiet,
            "debug": self.debug,
            "color": self.should_use_color(),
            "timestamps": self.show_timestamps,
            "rich": self.rich_output,
        }

    def get_markdown_config(self) -> Dict[str, Any]:
        """Get markdown export configuration.

        Returns:
            Markdown configuration dictionary
        """
        return {
            "include_timestamps": self.markdown_include_timestamps,
            "include_speakers": self.markdown_include_speakers,
            "include_confidence": self.markdown_include_confidence,
            "template": self.markdown_template,
        }

    def format_output(self, data: Any, format_override: Optional[OutputFormat] = None) -> str:
        """Format data for output based on configuration.

        Args:
            data: Data to format
            format_override: Optional format override

        Returns:
            Formatted string
        """
        output_format = format_override or self.output_format

        if output_format == OutputFormat.JSON:
            import json

            return json.dumps(data, indent=self.json_indent, sort_keys=self.json_sort_keys)

        elif output_format == OutputFormat.YAML:
            import yaml

            return yaml.dump(data, default_flow_style=False, sort_keys=self.json_sort_keys)

        elif output_format == OutputFormat.TABLE:
            # Would use tabulate or rich.table here
            return str(data)

        elif output_format == OutputFormat.MARKDOWN:
            # Format as markdown
            return str(data)

        elif output_format == OutputFormat.HTML:
            # Format as HTML
            return str(data)

        else:  # TEXT
            return str(data)

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for UI config
        """
        return ConfigurationSchema(
            name="UIConfig",
            required_fields=set(),
            optional_fields={
                "output_format": "text",
                "verbose": False,
                "quiet": False,
                "no_color": False,
                "show_progress": True,
                "rich_output": True,
                "interactive_mode": False,
            },
            validators={
                "output_format": lambda x: x in {f.value for f in OutputFormat},
                "color_scheme": lambda x: x in {c.value for c in ColorScheme},
                "progress_style": lambda x: x in {p.value for p in ProgressStyle},
                "json_indent": lambda x: isinstance(x, int) and 0 <= x <= 8,
                "progress_bar_width": lambda x: isinstance(x, int) and x > 0,
                "terminal_width": lambda x: isinstance(x, int) and x > 0,
                "terminal_height": lambda x: isinstance(x, int) and x > 0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "output_format": self.output_format.value,
            "verbose": self.verbose,
            "quiet": self.quiet,
            "debug": self.debug,
            "no_color": self.no_color,
            "force_color": self.force_color,
            "color_scheme": self.color_scheme.value,
            "theme": self.theme.name,
            "show_progress": self.show_progress,
            "progress_style": self.progress_style.value,
            "rich_output": self.rich_output,
            "interactive_mode": self.interactive_mode,
            "terminal_width": self.terminal_width,
            "terminal_height": self.terminal_height,
            "markdown_config": self.get_markdown_config(),
        }


# Singleton instance getter
def get_ui_config() -> UIConfig:
    """Get UI configuration instance.

    Returns:
        UIConfig singleton instance
    """
    return UIConfig.get_instance()
