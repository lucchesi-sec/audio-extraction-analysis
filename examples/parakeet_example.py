#!/usr/bin/env python3
"""
Example script demonstrating how to use the Parakeet transcription provider.
"""

import asyncio
from pathlib import Path
from src.providers.factory import TranscriptionProviderFactory


async def main():
    """Demonstrate Parakeet provider usage."""
    
    # Create Parakeet provider
    try:
        provider = TranscriptionProviderFactory.create_provider("parakeet")
        print(f"Created provider: {provider.get_provider_name()}")
    except Exception as e:
        print(f"Failed to create Parakeet provider: {e}")
        return

    # Check health
    try:
        health = await provider.health_check_async()
        print(f"Provider health: {health['status']}")
        if not health['healthy']:
            print("Provider is not healthy, exiting")
            return
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # For demonstration, you would need an actual audio file to transcribe
    # Example: audio_file_path = Path("path/to/your/audio/file.mp3")
    # Then call: result = await provider.transcribe_async(audio_file_path, language="en")

    print("Parakeet provider setup complete!")


if __name__ == "__main__":
    asyncio.run(main())