"""Security configuration and validation module."""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Pattern

from .base import BaseConfig, ConfigurationSchema

logger = logging.getLogger(__name__)


class SecurityConfig(BaseConfig):
    """Security settings and API key validation."""

    # API key format patterns
    _KEY_PATTERNS: Dict[str, Pattern] = {
        "deepgram": re.compile(r"^[a-f0-9]{40}$", re.IGNORECASE),
        "elevenlabs": re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE),
        "gemini": re.compile(r"^AIza[A-Za-z0-9_-]{35}$"),
        "openai": re.compile(r"^sk-[A-Za-z0-9]{48}$"),
        "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9]{95}$"),
    }

    # Rate limiting
    _rate_limits: Dict[str, Dict[str, Any]] = {}
    _rate_limit_lock = Lock()

    def __init__(self):
        """Initialize security configuration."""
        super().__init__()

        # API Keys (encrypted in memory if possible)
        self._api_keys: Dict[str, Optional[str]] = {
            "deepgram": self._load_api_key("DEEPGRAM_API_KEY"),
            "elevenlabs": self._load_api_key("ELEVENLABS_API_KEY"),
            "gemini": self._load_api_key("GEMINI_API_KEY"),
            "openai": self._load_api_key("OPENAI_API_KEY"),
            "anthropic": self._load_api_key("ANTHROPIC_API_KEY"),
        }

        # Security settings
        self.enable_api_key_validation = self.parse_bool(
            self.get_value("ENABLE_API_KEY_VALIDATION", "true")
        )
        self.enable_rate_limiting = self.parse_bool(self.get_value("ENABLE_RATE_LIMITING", "true"))
        self.enable_input_sanitization = self.parse_bool(
            self.get_value("ENABLE_INPUT_SANITIZATION", "true")
        )

        # Rate limiting settings
        self.rate_limit_window = int(self.get_value("RATE_LIMIT_WINDOW", "60"))  # seconds
        self.rate_limit_max_requests = int(self.get_value("RATE_LIMIT_MAX_REQUESTS", "100"))

        # File security
        self.max_path_length = int(self.get_value("MAX_PATH_LENGTH", "4096"))
        self.blocked_path_patterns = self.parse_list(
            self.get_value("BLOCKED_PATH_PATTERNS", "../,..\\,~/.,/etc/,/sys/,/proc/")
        )

        # Network security
        self.allowed_hosts = self.parse_list(self.get_value("ALLOWED_HOSTS", ""))
        self.ssl_verify = self.parse_bool(self.get_value("SSL_VERIFY", "true"))
        self.request_timeout = int(self.get_value("REQUEST_TIMEOUT", "30"))

        # Audit logging
        self.enable_audit_logging = self.parse_bool(self.get_value("ENABLE_AUDIT_LOGGING", "false"))
        self.audit_log_file = self.get_value("AUDIT_LOG_FILE", "audit.log")

        # Key rotation tracking
        self._key_rotation_tracking: Dict[str, datetime] = {}

    def _load_api_key(self, env_var: str) -> Optional[str]:
        """Load and optionally encrypt API key.

        Args:
            env_var: Environment variable name

        Returns:
            API key or None
        """
        key = self.get_value(env_var)
        if key:
            # In production, you might want to encrypt keys in memory
            # For now, we just store them as-is
            return key.strip()
        return None

    def validate_api_key(self, provider: str, key: Optional[str] = None) -> bool:
        """Validate API key format for a provider.

        Args:
            provider: Provider name
            key: API key to validate (if None, uses stored key)

        Returns:
            True if valid, False otherwise
        """
        if not self.enable_api_key_validation:
            return True

        if key is None:
            key = self._api_keys.get(provider)

        if not key:
            logger.warning(f"No API key configured for {provider}")
            return False

        if provider not in self._KEY_PATTERNS:
            logger.debug(f"No validation pattern for {provider}, assuming valid")
            return True

        pattern = self._KEY_PATTERNS[provider]
        if not pattern.match(key):
            logger.warning(
                f"API key format invalid for {provider}: {self.sanitize_for_logging(key)}"
            )
            return False

        # Log successful validation (audit)
        if self.enable_audit_logging:
            self._audit_log(f"API key validated for {provider}")

        return True

    def get_api_key(self, provider: str, validate: bool = True) -> str:
        """Get API key for a provider with optional validation.

        Args:
            provider: Provider name
            validate: Whether to validate the key

        Returns:
            API key

        Raises:
            ValueError: If key is missing or invalid
        """
        key = self._api_keys.get(provider)

        if not key:
            raise ValueError(
                f"API key for {provider} not configured. "
                f"Set {provider.upper()}_API_KEY environment variable."
            )

        if validate and not self.validate_api_key(provider, key):
            raise ValueError(f"API key for {provider} failed validation")

        # Track key usage for rotation monitoring
        self._track_key_usage(provider)

        return key

    def _track_key_usage(self, provider: str) -> None:
        """Track API key usage for rotation monitoring.

        Args:
            provider: Provider name
        """
        if provider not in self._key_rotation_tracking:
            self._key_rotation_tracking[provider] = datetime.now()

    def check_key_rotation_needed(self, provider: str, max_age_days: int = 90) -> bool:
        """Check if API key rotation is recommended.

        Args:
            provider: Provider name
            max_age_days: Maximum key age in days

        Returns:
            True if rotation is recommended
        """
        if provider not in self._key_rotation_tracking:
            return False

        key_age = datetime.now() - self._key_rotation_tracking[provider]
        return key_age > timedelta(days=max_age_days)

    def sanitize_path(self, path: Path) -> Path:
        """Sanitize file path for security.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path

        Raises:
            ValueError: If path is invalid or dangerous
        """
        if not self.enable_input_sanitization:
            return path

        # Convert to absolute path and resolve
        try:
            path = path.resolve()
        except Exception as e:
            raise ValueError(f"Invalid path: {e}")

        # Check path length
        if len(str(path)) > self.max_path_length:
            raise ValueError(f"Path too long: {len(str(path))} > {self.max_path_length}")

        # Check for blocked patterns
        path_str = str(path)
        for pattern in self.blocked_path_patterns:
            if pattern in path_str:
                raise ValueError(f"Path contains blocked pattern: {pattern}")

        return path

    def check_rate_limit(self, identifier: str, increment: bool = True) -> bool:
        """Check if rate limit is exceeded for an identifier.

        Args:
            identifier: Unique identifier (e.g., API key, IP address)
            increment: Whether to increment the counter

        Returns:
            True if within limits, False if exceeded
        """
        if not self.enable_rate_limiting:
            return True

        with self._rate_limit_lock:
            now = datetime.now()

            # Initialize or clean up old entries
            if identifier not in self._rate_limits:
                self._rate_limits[identifier] = {"requests": [], "window_start": now}

            # Clean old requests outside the window
            window_start = now - timedelta(seconds=self.rate_limit_window)
            self._rate_limits[identifier]["requests"] = [
                req_time
                for req_time in self._rate_limits[identifier]["requests"]
                if req_time > window_start
            ]

            # Check if limit exceeded
            current_count = len(self._rate_limits[identifier]["requests"])
            if current_count >= self.rate_limit_max_requests:
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False

            # Increment counter if requested
            if increment:
                self._rate_limits[identifier]["requests"].append(now)

            return True

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data for storage or comparison.

        Args:
            data: Data to hash
            salt: Optional salt value

        Returns:
            Hashed value
        """
        if salt:
            data = f"{salt}{data}"

        return hashlib.sha256(data.encode()).hexdigest()

    def _audit_log(self, message: str, level: str = "INFO") -> None:
        """Write to audit log.

        Args:
            message: Audit message
            level: Log level
        """
        if not self.enable_audit_logging:
            return

        timestamp = datetime.now().isoformat()
        audit_entry = f"[{timestamp}] [{level}] {message}\n"

        try:
            with open(self.audit_log_file, "a") as f:
                f.write(audit_entry)
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def validate_url(self, url: str) -> bool:
        """Validate URL against security policies.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic URL validation
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(url):
            logger.warning(f"Invalid URL format: {url}")
            return False

        # Check against allowed hosts if configured
        if self.allowed_hosts:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.hostname not in self.allowed_hosts:
                logger.warning(f"Host not in allowed list: {parsed.hostname}")
                return False

        return True

    def get_schema(self) -> ConfigurationSchema:
        """Get configuration schema.

        Returns:
            ConfigurationSchema for security config
        """
        return ConfigurationSchema(
            name="SecurityConfig",
            required_fields=set(),
            optional_fields={
                "enable_api_key_validation": True,
                "enable_rate_limiting": True,
                "enable_input_sanitization": True,
                "rate_limit_window": 60,
                "rate_limit_max_requests": 100,
                "ssl_verify": True,
                "request_timeout": 30,
            },
            validators={
                "rate_limit_window": lambda x: isinstance(x, int) and x > 0,
                "rate_limit_max_requests": lambda x: isinstance(x, int) and x > 0,
                "request_timeout": lambda x: isinstance(x, int) and x > 0,
                "max_path_length": lambda x: isinstance(x, int) and x > 0 and x <= 4096,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary (with sensitive data redacted)
        """
        return {
            "enable_api_key_validation": self.enable_api_key_validation,
            "enable_rate_limiting": self.enable_rate_limiting,
            "enable_input_sanitization": self.enable_input_sanitization,
            "rate_limit_window": self.rate_limit_window,
            "rate_limit_max_requests": self.rate_limit_max_requests,
            "max_path_length": self.max_path_length,
            "blocked_path_patterns": self.blocked_path_patterns,
            "allowed_hosts": self.allowed_hosts,
            "ssl_verify": self.ssl_verify,
            "request_timeout": self.request_timeout,
            "enable_audit_logging": self.enable_audit_logging,
            "audit_log_file": self.audit_log_file,
            # Redact API keys
            "api_keys_configured": {
                provider: key is not None for provider, key in self._api_keys.items()
            },
        }

    def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers for HTTP requests.

        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }


# Singleton instance getter
def get_security_config() -> SecurityConfig:
    """Get security configuration instance.

    Returns:
        SecurityConfig singleton instance
    """
    return SecurityConfig.get_instance()
