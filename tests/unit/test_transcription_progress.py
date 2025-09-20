"""Test for transcription service with progress functionality."""

from src.services.transcription import TranscriptionService


class TestTranscriptionServiceProgress:
    """Test transcription service with progress functionality."""

    def test_import_works(self):
        """Test that we can import the transcription service."""
        service = TranscriptionService()
        assert service is not None

    def test_transcribe_with_progress_method_exists(self):
        """Test that the progress transcription method exists."""
        service = TranscriptionService()
        assert hasattr(service, "transcribe_with_progress")

    def test_get_provider_speed_method_exists(self):
        """Test that the provider speed method exists."""
        service = TranscriptionService()
        assert hasattr(service, "_get_provider_speed")

    def test_calculate_sigmoid_progress_method_exists(self):
        """Test that the sigmoid progress method exists."""
        service = TranscriptionService()
        assert hasattr(service, "_calculate_sigmoid_progress")
