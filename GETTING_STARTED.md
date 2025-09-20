# Getting Started with Audio Extraction & Analysis

A powerful Python tool for extracting audio from video files and generating AI-powered transcriptions with comprehensive analysis.

## 🎯 What This Tool Does

This tool provides a complete pipeline for:
- **Audio Extraction**: Extract high-quality audio from video files using FFmpeg
- **AI Transcription**: Generate accurate transcriptions using Deepgram or ElevenLabs
- **Smart Analysis**: Produce structured analysis including summaries, chapters, and key insights
- **Flexible Output**: Support for multiple output formats (text, JSON, markdown)

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg installed on your system
- At least one API key (Deepgram or ElevenLabs)

### Installing FFmpeg

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

#### Verify Installation
```bash
ffmpeg -version
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lucchesi-sec/audio-extraction-analysis.git
cd audio-extraction-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

For development with testing capabilities:
```bash
pip install -e ".[dev]"
```

## 🔑 Configuration

### API Keys Setup

You'll need at least one transcription service API key:

#### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root:
```bash
# Deepgram (Recommended for longer files)
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# ElevenLabs (Great for shorter files, <10MB)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

#### Option 2: Export to Shell
```bash
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"
export ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"
```

### Getting API Keys

- **Deepgram**: Sign up at [console.deepgram.com](https://console.deepgram.com/)
  - Free tier includes $200 credit
  - No file size limits
  - Recommended for production use

- **ElevenLabs**: Sign up at [elevenlabs.io](https://elevenlabs.io/)
  - Free tier available
  - 10MB file size limit
  - Good for quick tests

## 🎮 Basic Usage

### Quick Start: Complete Pipeline

Process a video file from extraction to analysis in one command:

```bash
audio-extraction-analysis process video.mp4
```

This will:
1. Extract audio from the video
2. Transcribe using the best available provider
3. Generate comprehensive analysis
4. Save all outputs to `data/output/`

### Step-by-Step Commands

#### 1. Extract Audio Only
```bash
audio-extraction-analysis extract video.mp4 --output audio.mp3
```

Options:
- `--quality`: Choose quality preset (`speech`, `music`, `high`)
- `--output`: Specify output path (default: `data/output/`)

#### 2. Transcribe Audio
```bash
audio-extraction-analysis transcribe audio.mp3 --output transcript.txt
```

Options:
- `--provider`: Choose provider (`deepgram`, `elevenlabs`, or `auto`)
- `--language`: Specify language code (default: `en`)
- `--format`: Output format (`text`, `json`, `markdown`)

#### 3. Complete Pipeline with Options
```bash
audio-extraction-analysis process video.mp4 \
    --output-dir results/ \
    --quality speech \
    --provider deepgram \
    --format markdown \
    --verbose
```

## 📁 Project Structure

```
audio-extraction-analysis/
├── data/
│   ├── input/        # Place your video/audio files here
│   └── output/       # Generated transcripts and analysis
├── src/
│   ├── cli.py        # Command-line interface
│   ├── services/     # Core services (extraction, transcription)
│   ├── providers/    # Transcription provider implementations
│   └── analysis/     # Analysis engines
└── tests/            # Test suite
```

## 🎯 Common Workflows

### Podcast/Interview Processing
```bash
# Extract with speech-optimized settings
audio-extraction-analysis process podcast.mp4 \
    --quality speech \
    --provider deepgram \
    --format markdown
```

### Batch Processing
```bash
# Process multiple files
for file in data/input/*.mp4; do
    audio-extraction-analysis process "$file" \
        --output-dir "data/output/$(basename "$file" .mp4)/"
done
```

### Quick Transcription Test
```bash
# Test with a small audio file
audio-extraction-analysis transcribe sample.mp3 \
    --provider elevenlabs \
    --format json
```

## 📊 Output Formats

### Text Format
Simple plain text transcript, ideal for reading or further processing.

### JSON Format
Structured data including:
- Full transcript
- Timestamps
- Speaker identification (if available)
- Metadata (duration, word count, etc.)

### Markdown Export (New)
Generate professionally formatted Markdown transcripts with timestamps, speaker labels, and metadata.

Quick examples:
```bash
# Dedicated command
audio-extraction-analysis export-markdown audio.mp3 --output-dir ./output --template default

# Add to transcribe
audio-extraction-analysis transcribe audio.mp3 --export-markdown --md-template detailed

# Add to process
audio-extraction-analysis process video.mp4 --export-markdown --md-no-speakers
```

Output structure:
```
./output/
└── <source_name>/
    ├── transcript.md
    ├── metadata.json
    └── segments.json
```

### Markdown Format
Formatted document with:
- Executive summary
- Chapter breakdown
- Key topics and themes
- Full transcript with timestamps
- Key insights and takeaways

## 🎛️ Advanced Features

### Progress Tracking
All commands show real-time progress:
```
[Audio Extraction] ████████████████████ 100% | Extracting audio...
[Transcription]    ████████████████████ 100% | Processing...
[Analysis]         ████████████████████ 100% | Generating insights...
```

### JSON Output Mode
For CI/CD integration, use JSON output:
```bash
audio-extraction-analysis process video.mp4 --json
```

### Custom Provider Selection
The tool automatically selects the best provider, but you can override:
```bash
# Force specific provider
audio-extraction-analysis transcribe audio.mp3 --provider deepgram

# Auto-select based on file size and availability
audio-extraction-analysis transcribe audio.mp3 --provider auto
```

## 🐛 Troubleshooting

### Common Issues

#### "No transcription providers are configured"
**Solution**: Ensure you have set at least one API key in your `.env` file or environment variables.

#### "FFmpeg not found"
**Solution**: Install FFmpeg and ensure it's in your system PATH.

#### "File size exceeds limit for provider"
**Solution**: Use Deepgram for files larger than 10MB, or split the file.

#### Import errors
**Solution**: Ensure you've installed the package with `pip install -e .`

### Debug Mode
Run with verbose output for debugging:
```bash
audio-extraction-analysis --verbose process video.mp4
```

### Check Configuration
Verify your setup:
```bash
# Check FFmpeg
ffmpeg -version

# Check Python version
python --version

# Test import
python -c "from src.cli import main; print('Setup OK')"
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_audio_extraction.py
```

## 📚 Examples

Check the `examples/` directory for sample outputs:
- Executive summaries
- Chapter breakdowns
- Full transcripts with analysis

## 🤝 Getting Help

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/lucchesi-sec/audio-extraction-analysis/issues)
- **Documentation**: Check the `docs/` directory for detailed guides
- **API Documentation**: Run `audio-extraction-analysis --help` for CLI reference

## 🚦 Next Steps

1. **Try the Quick Start**: Process your first video with the `process` command
2. **Explore Output Formats**: Experiment with different `--format` options
3. **Optimize Quality**: Test different `--quality` settings for your content type
4. **Batch Processing**: Set up scripts for processing multiple files
5. **Integration**: Use JSON output mode for integration with other tools

## 📄 License

MIT License - See LICENSE file for details

---

**Pro Tips:**
- 🎯 Use `--quality speech` for podcasts and interviews
- 🚀 Deepgram is faster and handles larger files
- 📁 Keep input files in `data/input/` - they're automatically git-ignored
- 💡 Use `--verbose` to understand what's happening under the hood
- 🔄 The `process` command is usually all you need for most use cases
