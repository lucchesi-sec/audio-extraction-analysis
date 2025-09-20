"""Test for async audio extraction functionality."""

import pytest

from src.services.audio_extraction_async import AsyncAudioExtractor


class TestAsyncAudioExtractor:
    """Test async audio extraction functionality."""

    def test_import_works(self):
        """Test that we can import the async audio extractor."""
        extractor = AsyncAudioExtractor()
        assert extractor is not None

    @pytest.mark.asyncio
    async def test_extract_audio_async_method_exists(self):
        """Test that the async extraction method exists."""
        extractor = AsyncAudioExtractor()
        assert hasattr(extractor, "extract_audio_async")
