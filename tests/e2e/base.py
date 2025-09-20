"""
Base classes and utilities for end-to-end testing.
"""
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class TestResult:
    """Encapsulates test execution results."""
    success: bool
    output: str
    error: str
    duration: float
    exit_code: int
    
    @property
    def failed(self) -> bool:
        return not self.success


@dataclass
class TestFile:
    """Test media file specification."""
    name: str
    path: Path
    duration: Optional[float] = None
    size: Optional[int] = None
    format: Optional[str] = None
    description: str = ""


class E2ETestBase:
    """Base class for end-to-end tests with common utilities."""

    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="e2e_test_"))
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Store original environment
        self.original_env = dict(os.environ)

        # Ensure shared test data is available
        if not hasattr(self, "test_data_manager"):
            from .test_data_manager import TestDataManager

            self.test_data_manager = TestDataManager()
        if not hasattr(self, "test_files"):
            self.test_files = self.test_data_manager.generate_all_test_files()
        
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Cleanup temporary files
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove temporary test files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_cli_command(
        self, 
        command: List[str], 
        timeout: int = 300,
        env_vars: Optional[Dict[str, str]] = None
    ) -> TestResult:
        """
        Execute CLI command and return results.
        
        Args:
            command: CLI command as list of strings
            timeout: Maximum execution time in seconds
            env_vars: Additional environment variables
            
        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        
        # Prepare environment
        test_env = os.environ.copy()
        if env_vars:
            test_env.update(env_vars)
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=test_env,
                cwd=self.temp_dir
            )
            
            duration = time.time() - start_time
            
            return TestResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                duration=duration,
                exit_code=-1
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                success=False,
                output="",
                error=str(e),
                duration=duration,
                exit_code=-1
            )
    
    def validate_output_files(self, expected_files: List[str]) -> Dict[str, bool]:
        """
        Validate that expected output files exist and are non-empty.
        
        Args:
            expected_files: List of expected output file names
            
        Returns:
            Dictionary mapping file names to existence status
        """
        results = {}
        for filename in expected_files:
            file_path = self.output_dir / filename
            exists = file_path.exists()
            non_empty = exists and file_path.stat().st_size > 0
            results[filename] = exists and non_empty
        return results
    
    def assert_files_exist(self, expected_files: List[str]):
        """Assert that all expected files exist and are non-empty."""
        validation_results = self.validate_output_files(expected_files)
        for filename, exists in validation_results.items():
            assert exists, f"Expected output file '{filename}' does not exist or is empty"
    
    def set_test_env(self, **env_vars):
        """Set environment variables for testing."""
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def get_test_data_path(self) -> Path:
        """Get path to test data directory."""
        return Path(__file__).parent / "test_data"

    # Pytest-based suite shim to support unittest-style subTest contexts used in legacy tests
    @contextmanager
    def subTest(self, **_params):
        yield


class CLITestMixin:
    """Mixin for CLI-specific test utilities."""
    
    def run_extract_command(
        self, 
        input_file: Union[str, Path], 
        quality: str = "high",
        output_file: Optional[Union[str, Path]] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run audio extraction command."""
        cmd = ["audio-extraction-analysis", "extract", str(input_file), "--quality", quality]
        
        if output_file:
            cmd.extend(["--output", str(output_file)])
        
        if additional_args:
            cmd.extend(additional_args)
        
        return self.run_cli_command(cmd)
    
    def run_transcribe_command(
        self,
        input_file: Union[str, Path],
        provider: str = "auto",
        language: str = "en",
        output_file: Optional[Union[str, Path]] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run transcription command."""
        cmd = [
            "audio-extraction-analysis", "transcribe", str(input_file),
            "--provider", provider,
            "--language", language
        ]
        
        if output_file:
            cmd.extend(["--output", str(output_file)])
        
        if additional_args:
            cmd.extend(additional_args)
        
        return self.run_cli_command(cmd)
    
    def run_process_command(
        self,
        input_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        provider: str = "auto",
        quality: str = "high",
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run full processing pipeline command."""
        cmd = [
            "audio-extraction-analysis", "process", str(input_file),
            "--provider", provider,
            "--quality", quality
        ]
        
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
        
        if additional_args:
            cmd.extend(additional_args)
        
        return self.run_cli_command(cmd)


class PerformanceTestMixin:
    """Mixin for performance testing utilities."""
    
    def measure_execution_time(self, func, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    def assert_performance_target(self, duration: float, target: float, operation: str):
        """Assert that operation meets performance target."""
        assert duration <= target, f"{operation} took {duration:.2f}s, expected <= {target}s"
    
    def monitor_memory_usage(self):
        """Context manager for monitoring memory usage during test."""
        import os

        import psutil
        
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.peak_memory = 0
                self.initial_memory = 0
            
            def __enter__(self):
                self.initial_memory = self.process.memory_info().rss
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            
            def update_peak(self):
                current_memory = self.process.memory_info().rss
                self.peak_memory = max(self.peak_memory, current_memory)
            
            def get_memory_usage_mb(self):
                return (self.peak_memory - self.initial_memory) / 1024 / 1024
        
        return MemoryMonitor()


class SecurityTestMixin:
    """Mixin for security testing utilities."""
    
    def create_malicious_filename(self, attack_type: str) -> str:
        """Create filenames for security testing."""
        attacks = {
            "path_traversal": "../../../etc/passwd",
            "command_injection": "file.mp4; rm -rf /",
            "unicode": "test_unicode_名前.mp4",
            "spaces": "test file with spaces.mp4",
            "special_chars": "test@#$%^&*()file.mp4",
            "long_name": "a" * 300 + ".mp4"
        }
        return attacks.get(attack_type, "safe_test.mp4")
    
    def validate_output_sanitization(self, output: str) -> Dict[str, bool]:
        """Validate that output is properly sanitized."""
        checks = {
            "no_control_chars": not any(ord(c) < 32 for c in output if c not in '\n\r\t'),
            "no_script_tags": "<script>" not in output.lower(),
            "no_api_keys": not self._contains_api_key_pattern(output),
            "reasonable_length": len(output) < 1000000  # 1MB max
        }
        return checks
    
    def _contains_api_key_pattern(self, text: str) -> bool:
        """Check if text contains potential API key patterns."""
        import re
        # Common API key patterns
        patterns = [
            r'[A-Za-z0-9]{32,}',  # Long alphanumeric strings
            r'sk-[A-Za-z0-9]{48}',  # OpenAI-style keys
            r'[A-Za-z0-9+/]{40,}={0,2}',  # Base64-encoded keys
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False


class MockProviderMixin:
    """Mixin for mocking transcription providers during testing."""
    
    def mock_successful_transcription(self):
        """Mock a successful transcription response."""
        return {
            "transcript": "This is a test transcription.",
            "speakers": [
                {"speaker": 0, "text": "Hello, this is speaker one."},
                {"speaker": 1, "text": "And this is speaker two responding."}
            ],
            "duration": 30.0,
            "confidence": 0.95
        }
    
    def mock_provider_failure(self, error_type: str = "api_error"):
        """Mock various provider failure scenarios."""
        errors = {
            "api_error": "API service temporarily unavailable",
            "auth_error": "Invalid API key",
            "file_too_large": "File exceeds maximum size limit",
            "unsupported_format": "Audio format not supported",
            "rate_limit": "Rate limit exceeded"
        }
        return {"error": errors.get(error_type, "Unknown error")}
