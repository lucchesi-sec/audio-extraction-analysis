#!/usr/bin/env python3
"""
Test script to demonstrate the Parakeet transcription provider.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import torchaudio
import torch
from src.providers.factory import TranscriptionProviderFactory


async def create_test_audio():
    """Create a simple test audio file for transcription."""
    # Create a simple sine wave tone as test audio
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Generate time vector
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave
    waveform = torch.sin(2 * torch.pi * frequency * t)
    
    # Add some noise to make it more interesting
    noise = torch.randn_like(waveform) * 0.1
    waveform = waveform + noise
    
    # Normalize
    waveform = waveform / torch.max(torch.abs(waveform))
    
    # Add channel dimension (mono)
    waveform = waveform.unsqueeze(0)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        torchaudio.save(tmp_file.name, waveform, sample_rate)
        return Path(tmp_file.name)


async def main():
    """Test the Parakeet provider with a simple audio file."""
    print("Testing Parakeet transcription provider...")
    
    # Create test audio
    print("Creating test audio file...")
    audio_path = await create_test_audio()
    
    try:
        # Create Parakeet provider
        print("Creating Parakeet provider...")
        provider = TranscriptionProviderFactory.create_provider("parakeet")
        print(f"Provider: {provider.get_provider_name()}")
        print(f"Supported features: {provider.get_supported_features()}")
        
        # Check health
        print("Checking provider health...")
        health = await provider.health_check_async()
        print(f"Health status: {health['status']}")
        if health['healthy']:
            print("Provider is healthy and ready!")
        else:
            print(f"Provider is not healthy: {health['details']}")
            return
        
        # Note: For a real test, we would transcribe an actual speech audio file
        # Since we created a sine wave tone, it won't produce meaningful transcription
        # But we can at least verify the provider is working correctly
        print(f"Test audio file created at: {audio_path}")
        print("Note: This is a sine wave tone, not speech, so transcription will not be meaningful.")
        print("In a real scenario, you would use an actual speech audio file.")
        
    finally:
        # Clean up test audio file
        if audio_path.exists():
            audio_path.unlink()
            print(f"Cleaned up test audio file: {audio_path}")


if __name__ == "__main__":
    asyncio.run(main())