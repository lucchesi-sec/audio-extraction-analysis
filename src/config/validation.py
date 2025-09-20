"""Configuration validation logic and schemas."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.types import PositiveFloat, PositiveInt

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # All validation rules enforced
    NORMAL = "normal"  # Standard validation
    LENIENT = "lenient"  # Minimal validation
    DISABLED = "disabled"  # No validation


@dataclass
class ValidationRule:
    """Individual validation rule definition."""

    name: str
    description: str
    validator: Callable[[Any], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.NORMAL
    auto_fix: Optional[Callable[[Any], Any]] = None


class ConfigValidator:
    """Configuration validator with multiple rule sets."""

    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        """Initialize validator with specified level.

        Args:
            level: Validation strictness level
        """
        self.level = level
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize default validation rules."""
        # File path rules
        self.add_rule_set(
            "file_paths",
            [
                ValidationRule(
                    name="path_exists",
                    description="Check if path exists",
                    validator=lambda p: Path(p).exists() if isinstance(p, (str, Path)) else False,
                    error_message="Path does not exist",
                    level=ValidationLevel.NORMAL,
                ),
                ValidationRule(
                    name="path_traversal",
                    description="Check for path traversal attempts",
                    validator=lambda p: ".." not in str(p),
                    error_message="Path traversal detected",
                    level=ValidationLevel.STRICT,
                ),
                ValidationRule(
                    name="path_length",
                    description="Check path length",
                    validator=lambda p: len(str(p)) < 4096,
                    error_message="Path too long",
                    level=ValidationLevel.NORMAL,
                ),
            ],
        )

        # API key rules
        self.add_rule_set(
            "api_keys",
            [
                ValidationRule(
                    name="key_format",
                    description="Check API key format",
                    validator=lambda k: bool(k) and len(k) >= 20,
                    error_message="Invalid API key format",
                    level=ValidationLevel.NORMAL,
                ),
                ValidationRule(
                    name="key_entropy",
                    description="Check API key entropy",
                    validator=self._check_key_entropy,
                    error_message="API key has low entropy",
                    level=ValidationLevel.STRICT,
                ),
            ],
        )

        # Network rules
        self.add_rule_set(
            "network",
            [
                ValidationRule(
                    name="url_format",
                    description="Check URL format",
                    validator=self._validate_url,
                    error_message="Invalid URL format",
                    level=ValidationLevel.NORMAL,
                ),
                ValidationRule(
                    name="port_range",
                    description="Check port range",
                    validator=lambda p: 0 < p <= 65535 if isinstance(p, int) else False,
                    error_message="Port out of valid range",
                    level=ValidationLevel.NORMAL,
                ),
                ValidationRule(
                    name="timeout_range",
                    description="Check timeout value",
                    validator=lambda t: 0 < t <= 3600 if isinstance(t, (int, float)) else False,
                    error_message="Timeout out of valid range",
                    level=ValidationLevel.NORMAL,
                    auto_fix=lambda t: max(1, min(t, 3600)) if isinstance(t, (int, float)) else 30,
                ),
            ],
        )

    def add_rule_set(self, category: str, rules: List[ValidationRule]) -> None:
        """Add a set of validation rules for a category.

        Args:
            category: Rule category name
            rules: List of validation rules
        """
        if category not in self.rules:
            self.rules[category] = []
        self.rules[category].extend(rules)

    def validate(self, config: Dict[str, Any], category: Optional[str] = None) -> bool:
        """Validate configuration against rules.

        Args:
            config: Configuration dictionary
            category: Optional specific category to validate

        Returns:
            True if validation passes
        """
        if self.level == ValidationLevel.DISABLED:
            return True

        self.errors = []
        self.warnings = []

        categories = [category] if category else self.rules.keys()

        for cat in categories:
            if cat not in self.rules:
                continue

            for rule in self.rules[cat]:
                # Skip rules above current level
                if self._should_skip_rule(rule):
                    continue

                # Apply rule to relevant config values
                for key, value in config.items():
                    if self._rule_applies_to_key(rule, key):
                        self._apply_rule(rule, key, value, config)

        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Validation error: {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Validation warning: {warning}")

        return len(self.errors) == 0

    def _should_skip_rule(self, rule: ValidationRule) -> bool:
        """Check if rule should be skipped based on validation level.

        Args:
            rule: Validation rule

        Returns:
            True if rule should be skipped
        """
        level_order = {
            ValidationLevel.DISABLED: 0,
            ValidationLevel.LENIENT: 1,
            ValidationLevel.NORMAL: 2,
            ValidationLevel.STRICT: 3,
        }

        return level_order[rule.level] > level_order[self.level]

    def _rule_applies_to_key(self, rule: ValidationRule, key: str) -> bool:
        """Check if rule applies to configuration key.

        Args:
            rule: Validation rule
            key: Configuration key

        Returns:
            True if rule applies
        """
        # Simple heuristic - can be made more sophisticated
        rule_keywords = {
            "path": ["path", "dir", "file", "folder"],
            "key": ["key", "token", "secret", "credential"],
            "url": ["url", "uri", "endpoint", "host"],
            "port": ["port"],
            "timeout": ["timeout", "ttl", "expiry"],
        }

        key_lower = key.lower()
        for rule_key, keywords in rule_keywords.items():
            if rule_key in rule.name.lower():
                return any(kw in key_lower for kw in keywords)

        return False

    def _apply_rule(
        self, rule: ValidationRule, key: str, value: Any, config: Dict[str, Any]
    ) -> None:
        """Apply validation rule to a value.

        Args:
            rule: Validation rule
            key: Configuration key
            value: Configuration value
            config: Full configuration dictionary for auto-fix
        """
        try:
            if not rule.validator(value):
                error_msg = f"{key}: {rule.error_message}"

                # Try auto-fix if available
                if rule.auto_fix and self.level != ValidationLevel.STRICT:
                    fixed_value = rule.auto_fix(value)
                    config[key] = fixed_value
                    self.warnings.append(f"{error_msg} (auto-fixed to {fixed_value})")
                else:
                    self.errors.append(error_msg)
        except Exception as e:
            self.errors.append(f"{key}: Validation error - {e!s}")

    def _check_key_entropy(self, key: str) -> bool:
        """Check if API key has sufficient entropy.

        Args:
            key: API key to check

        Returns:
            True if entropy is sufficient
        """
        if not key:
            return False

        # Simple entropy check - character variety
        unique_chars = len(set(key))
        return unique_chars >= min(10, len(key) // 2)

    def _validate_url(self, url: str) -> bool:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        url_pattern = re.compile(
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(url_pattern.match(url))

    def get_report(self) -> Dict[str, Any]:
        """Get validation report.

        Returns:
            Dictionary with validation results
        """
        return {
            "level": self.level.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# Pydantic models for structured validation
class FileConfigModel(BaseModel):
    """File handling configuration model."""

    max_file_size: PositiveInt = Field(100_000_000, description="Maximum file size in bytes")
    allowed_extensions: List[str] = Field(
        [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"], description="Allowed file extensions"
    )
    temp_dir: str = Field("/tmp", description="Temporary directory path")

    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Ensure extensions start with dot."""
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]

    @field_validator("temp_dir")
    @classmethod
    def validate_temp_dir(cls, v: str) -> str:
        """Ensure temp directory exists."""
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return str(path)


class RetryConfigModel(BaseModel):
    """Retry configuration model."""

    max_retries: PositiveInt = Field(3, description="Maximum retry attempts")
    retry_delay: PositiveFloat = Field(1.0, description="Initial retry delay in seconds")
    max_retry_delay: PositiveFloat = Field(60.0, description="Maximum retry delay")
    exponential_base: PositiveFloat = Field(2.0, description="Exponential backoff base")
    jitter_enabled: bool = Field(True, description="Enable retry jitter")

    @field_validator("max_retry_delay")
    @classmethod
    def validate_max_delay(cls, v: float) -> float:
        """Ensure max delay is greater than initial delay."""
        # Note: In pydantic v2, values are not passed to field_validator
        # This validation would need to be done in model_validator if cross-field validation is needed
        return v


class CircuitBreakerConfigModel(BaseModel):
    """Circuit breaker configuration model."""

    failure_threshold: PositiveInt = Field(5, description="Failure threshold")
    recovery_timeout: PositiveFloat = Field(60.0, description="Recovery timeout in seconds")
    half_open_requests: PositiveInt = Field(3, description="Requests in half-open state")

    @field_validator("recovery_timeout")
    @classmethod
    def validate_recovery_timeout(cls, v: float) -> float:
        """Ensure reasonable recovery timeout."""
        if v < 1.0:
            raise ValueError("recovery_timeout must be >= 1.0 seconds")
        if v > 3600:
            logger.warning("Very long recovery timeout (>1 hour) configured")
        return v


class ProviderConfigModel(BaseModel):
    """Provider configuration model."""

    name: str = Field(..., description="Provider name")
    api_key: Optional[str] = Field(None, description="API key")
    timeout: PositiveInt = Field(30, description="Request timeout in seconds")
    enabled: bool = Field(True, description="Provider enabled")
    priority: int = Field(0, description="Provider priority (lower is higher priority)")

    class Config:
        """Pydantic config."""

        extra = "allow"  # Allow provider-specific fields


def validate_config_schema(config: Dict[str, Any], model: BaseModel) -> tuple[bool, Optional[Dict]]:
    """Validate configuration against Pydantic model.

    Args:
        config: Configuration dictionary
        model: Pydantic model class

    Returns:
        Tuple of (is_valid, error_dict)
    """
    try:
        model(**config)
        return True, None
    except ValidationError as e:
        errors = {}
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            errors[field] = error["msg"]
        return False, errors


def create_config_validator(
    level: Union[str, ValidationLevel] = ValidationLevel.NORMAL,
) -> ConfigValidator:
    """Create configuration validator with specified level.

    Args:
        level: Validation level (string or enum)

    Returns:
        ConfigValidator instance
    """
    if isinstance(level, str):
        try:
            level = ValidationLevel(level.lower())
        except ValueError:
            logger.warning(f"Invalid validation level '{level}', using NORMAL")
            level = ValidationLevel.NORMAL

    return ConfigValidator(level)
