"""Concise transcript analyzer that generates a single comprehensive analysis file."""
from __future__ import annotations

import logging
from pathlib import Path

from ..models.transcription import TranscriptionResult

logger = logging.getLogger(__name__)


class ConciseAnalyzer:
    """Generates a single, comprehensive analysis file from transcript data."""

    def __init__(self):
        """Initialize the concise analyzer."""
        pass

    def analyze_and_save(
        self, result: TranscriptionResult, output_dir: Path, filename_base: str
    ) -> Path:
        """Analyze transcript and save a single comprehensive analysis file.

        Args:
            result: TranscriptionResult with all the rich data from Deepgram
            output_dir: Directory to save the analysis file
            filename_base: Base filename (without extension)

        Returns:
            Path to the generated analysis file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = output_dir / f"{filename_base}_analysis.md"

        # Generate the analysis content
        content = self._generate_analysis(result)

        # Write to file
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Concise analysis saved to: {analysis_path}")
        return analysis_path

    def _generate_analysis(self, result: TranscriptionResult) -> str:
        """Generate the complete analysis content.

        Args:
            result: TranscriptionResult with rich transcript data

        Returns:
            Complete markdown analysis content
        """
        sections = [
            self._generate_header(result),
            self._generate_overview(result),
            self._generate_key_topics(result),
            self._generate_speaker_insights(result),
            self._generate_sentiment_analysis(result),
            self._generate_highlights_and_quotes(result),
            self._generate_action_items_and_intents(result),
            self._generate_timeline(result),
            self._generate_metadata(result),
        ]

        return "\n\n".join(filter(None, sections))

    def _generate_header(self, result: TranscriptionResult) -> str:
        """Generate the document header."""
        duration_formatted = self._format_duration(result.duration)

        return f"""# Audio Analysis Report

**Generated:** {result.generated_at.strftime("%Y-%m-%d %H:%M:%S")}  
**Provider:** {result.provider_name}  
**Duration:** {duration_formatted}  
**File:** {Path(result.audio_file).name}

---"""

    def _generate_overview(self, result: TranscriptionResult) -> str:
        """Generate overview section."""
        word_count = len(result.transcript.split())

        overview = f"""## 📋 Overview

**Content Length:** {len(result.transcript):,} characters / {word_count:,} words"""

        if result.summary:
            overview += f"\n\n**Summary:** {result.summary}"
        else:
            # Generate a simple summary from the first few sentences
            sentences = result.transcript.split(". ")[:3]
            simple_summary = (
                ". ".join(sentences) + "..."
                if len(sentences) >= 3
                else result.transcript[:300] + "..."
            )
            overview += f"\n\n**Key Content:** {simple_summary}"

        return overview

    def _generate_key_topics(self, result: TranscriptionResult) -> str:
        """Generate key topics section."""
        if not result.topics:
            return "## 🎯 Key Topics\n\n*No specific topics identified*"

        content = "## 🎯 Key Topics\n\n"

        # Sort topics by frequency
        sorted_topics = sorted(result.topics.items(), key=lambda x: x[1], reverse=True)

        for i, (topic, count) in enumerate(sorted_topics[:10], 1):
            content += f"{i}. **{topic.title()}** ({count} mentions)\n"

        return content

    def _generate_speaker_insights(self, result: TranscriptionResult) -> str:
        """Generate speaker insights section."""
        if not result.speakers:
            return "## 👥 Speaker Analysis\n\n*No speaker diarization data available*"

        content = "## 👥 Speaker Analysis\n\n"

        for speaker in result.speakers:
            percentage = (speaker.total_time / result.duration * 100) if result.duration > 0 else 0
            speaker_name = f"Speaker {speaker.id}"
            duration = self._format_duration(speaker.total_time)
            content += f"**{speaker_name}:** {duration} ({percentage:.1f}%)\n"

        return content

    def _generate_sentiment_analysis(self, result: TranscriptionResult) -> str:
        """Generate sentiment analysis section."""
        if not result.sentiment_distribution:
            return "## 😊 Sentiment Analysis\n\n*No sentiment analysis data available*"

        content = "## 😊 Sentiment Analysis\n\n"

        total_segments = sum(result.sentiment_distribution.values())
        for sentiment, count in result.sentiment_distribution.items():
            percentage = (count / total_segments * 100) if total_segments > 0 else 0
            emoji = self._get_sentiment_emoji(sentiment)
            content += f"{emoji} **{sentiment.title()}:** {count} segments ({percentage:.1f}%)\n"

        return content

    def _generate_highlights_and_quotes(self, result: TranscriptionResult) -> str:
        """Generate highlights and notable quotes section."""
        content = "## 💡 Key Highlights & Quotes\n\n"

        # Extract meaningful sentences (longer sentences often contain more substance)
        sentences = [s.strip() for s in result.transcript.split(".") if s.strip()]
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 15 and len(s) <= 200]

        # Take top sentences by length (simple heuristic)
        top_sentences = sorted(meaningful_sentences, key=len, reverse=True)[:5]

        if top_sentences:
            for i, sentence in enumerate(top_sentences, 1):
                content += f'{i}. *"{sentence.strip()}."\n\n'
        else:
            content += "*No specific highlights identified*\n"

        return content

    def _generate_action_items_and_intents(self, result: TranscriptionResult) -> str:
        """Generate action items and intents section."""
        content = "## ✅ Intents & Action Items\n\n"

        if result.intents:
            content += "**Identified Intents:**\n"
            for intent in set(result.intents):  # Remove duplicates
                content += f"- {intent.title()}\n"
        else:
            # Look for action-oriented language in the transcript
            action_keywords = [
                "should",
                "need to",
                "must",
                "will",
                "plan to",
                "going to",
                "have to",
            ]
            action_sentences = []

            for sentence in result.transcript.split("."):
                sentence = sentence.strip()
                if (
                    any(keyword in sentence.lower() for keyword in action_keywords)
                    and len(sentence) > 20
                ):
                    action_sentences.append(sentence)

            if action_sentences:
                content += "**Potential Action Items:**\n"
                for sentence in action_sentences[:5]:  # Limit to 5
                    content += f"- {sentence.strip()}\n"
            else:
                content += "*No specific action items or intents identified*"

        return content

    def _generate_timeline(self, result: TranscriptionResult) -> str:
        """Generate timeline section from chapters or utterances."""
        if not result.chapters and not result.utterances:
            return "## ⏰ Timeline\n\n*No timeline data available*"

        content = "## ⏰ Timeline\n\n"

        if result.chapters:
            content += "**Topic Segments:**\n"
            for chapter in result.chapters:
                start_time = self._format_duration(chapter.start_time)
                topics = ", ".join(chapter.topics) if chapter.topics else "General Discussion"
                content += f"- **{start_time}:** {topics}\n"
        elif result.utterances:
            content += "**Key Moments:**\n"
            # Show first few utterances as examples
            for utterance in result.utterances[:10]:
                start_time = self._format_duration(utterance.start)
                preview = (
                    utterance.text[:100] + "..." if len(utterance.text) > 100 else utterance.text
                )
                content += f"- **{start_time}:** {preview}\n"

        return content

    def _generate_metadata(self, result: TranscriptionResult) -> str:
        """Generate metadata section."""
        content = "## 📊 Technical Metadata\n\n"
        content += f"**Provider Features:** {', '.join(result.provider_features or [])}\n"
        content += f"**Audio File:** `{result.audio_file}`\n"
        content += f"**Processing Time:** {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if result.speakers:
            content += f"**Speakers Detected:** {len(result.speakers)}\n"

        if result.chapters:
            content += f"**Topic Segments:** {len(result.chapters)}\n"

        if result.utterances:
            content += f"**Total Utterances:** {len(result.utterances)}\n"

        return content

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _get_sentiment_emoji(self, sentiment: str) -> str:
        """Get emoji for sentiment."""
        sentiment_map = {"positive": "😊", "negative": "😔", "neutral": "😐"}
        return sentiment_map.get(sentiment.lower(), "🤔")
