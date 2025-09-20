"""Full transcript analyzer that generates 5 structured markdown files.

This analyzer produces the following files in the output directory:
1. 01_executive_summary.md
2. 02_chapter_overview.md
3. 03_key_topics_and_intents.md
4. 04_full_transcript_with_timestamps.md
5. 05_key_insights_and_takeaways.md

The structure broadly follows docs/transcription_formatting_prompts.md
and examples in the examples/ and data/output/ folders.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.transcription import TranscriptionResult, TranscriptionUtterance

logger = logging.getLogger(__name__)


class FullAnalyzer:
    """Generates 5 detailed analysis files from a TranscriptionResult."""

    def __init__(self) -> None:
        pass

    def analyze_and_save(
        self, result: TranscriptionResult, output_dir: Path, filename_base: str
    ) -> dict[str, Path]:
        """Generate all 5 analysis files and return their paths.

        Args:
            result: Rich transcription result
            output_dir: Folder to write files
            filename_base: Base name to include in metadata or future use

        Returns:
            Mapping of logical names to created file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "executive_summary": output_dir / "01_executive_summary.md",
            "chapter_overview": output_dir / "02_chapter_overview.md",
            "topics_intents": output_dir / "03_key_topics_and_intents.md",
            "full_transcript": output_dir / "04_full_transcript_with_timestamps.md",
            "key_insights": output_dir / "05_key_insights_and_takeaways.md",
        }

        paths["executive_summary"].write_text(
            self._render_executive_summary(result), encoding="utf-8"
        )
        paths["chapter_overview"].write_text(
            self._render_chapter_overview(result), encoding="utf-8"
        )
        paths["topics_intents"].write_text(
            self._render_topics_and_intents(result), encoding="utf-8"
        )
        paths["full_transcript"].write_text(self._render_full_transcript(result), encoding="utf-8")
        paths["key_insights"].write_text(self._render_key_insights(result), encoding="utf-8")

        logger.info("Full analysis generated: 5 markdown files")
        return paths

    # ---------------------- Renderers ----------------------
    def _render_executive_summary(self, result: TranscriptionResult) -> str:
        duration = self._format_hms(result.duration)
        speakers_count = len(result.speakers) if result.speakers else 0
        topic_count = len(result.topics or {})
        intents_count = len(result.intents or [])
        sentiment_overall = self._overall_sentiment(result)

        summary_text = result.summary or self._fallback_summary(result.transcript)

        return (
            f"# Executive Summary\n\n"
            f"## Session Information\n"
            f"- **Date Generated:** {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **Duration:** {duration}\n"
            f"- **Total Speakers:** {speakers_count}\n"
            f"- **Provider:** {result.provider_name}\n"
            f"- **Audio File:** {Path(result.audio_file).name}\n\n"
            f"## Executive Summary\n"
            f"{summary_text}\n\n"
            f"## Session Structure\n"
            f"- Total Chapters: {len(result.chapters or [])}\n"
            f"- Key Topics Discussed: {topic_count}\n"
            f"- Detected Intents: {intents_count}\n"
            f"- Overall Sentiment: {sentiment_overall}\n\n"
            f"## Quick Links\n"
            f"- [Chapter Overview](02_chapter_overview.md)\n"
            f"- [Topics & Intents](03_key_topics_and_intents.md)\n"
            f"- [Full Transcript](04_full_transcript_with_timestamps.md)\n"
            f"- [Key Insights](05_key_insights_and_takeaways.md)\n"
        )

    def _render_chapter_overview(self, result: TranscriptionResult) -> str:
        if not result.chapters:
            return "# Chapter-by-Chapter Overview\n\n_No chapters identified._\n"

        lines: list[str] = ["# Chapter-by-Chapter Overview", ""]
        total_duration = max(result.duration, 1e-6)

        for idx, ch in enumerate(result.chapters, 1):
            start = self._format_hms(ch.start_time)
            end = self._format_hms(ch.end_time)
            pct = (
                ((ch.end_time - ch.start_time) / total_duration) * 100
                if ch.end_time >= ch.start_time
                else 0.0
            )
            title = ", ".join(ch.topics) if ch.topics else f"Chapter {idx}"

            lines.append(f"## Chapter {idx}: {title}")
            lines.append(f"**Time:** [{start}] - [{end}] ({pct:.1f}% of session)")
            lines.append("")
            lines.append("### Topics Covered:")
            if ch.topics:
                for t in ch.topics:
                    lines.append(f"- {t}")
            else:
                lines.append("- General discussion")
            lines.append("\n---\n")

        # Optional summary table
        lines.append("\n## Chapter Summary Table\n")
        lines.append("| # | Time Range | % | Title |")
        lines.append("|---|------------|---:|-------|")
        for idx, ch in enumerate(result.chapters, 1):
            start = self._format_hms(ch.start_time)
            end = self._format_hms(ch.end_time)
            pct = (
                ((ch.end_time - ch.start_time) / total_duration) * 100
                if ch.end_time >= ch.start_time
                else 0.0
            )
            title = ", ".join(ch.topics) if ch.topics else f"Chapter {idx}"
            lines.append(f"| {idx} | [{start}] - [{end}] | {pct:.1f}% | {title} |")

        return "\n".join(lines) + "\n"

    def _render_topics_and_intents(self, result: TranscriptionResult) -> str:
        topics = result.topics or {}
        intents = result.intents or []
        sentiments = result.sentiment_distribution or {}

        lines: list[str] = ["# Key Topics and Detected Intents", ""]

        # Topic frequency table
        lines.append("## Topic Frequency Analysis")
        if topics:
            lines.append("| Topic | Mentions |")
            lines.append("|-------|----------:|")
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {topic} | {count} |")
        else:
            lines.append("_No specific topics identified._")

        # Intents
        lines.append("\n## Detected Intents")
        if intents:
            lines.append("| Intent |")
            lines.append("|--------|")
            for intent in sorted(set(intents)):
                lines.append(f"| {intent} |")
        else:
            lines.append("_No intents detected._")

        # Sentiment
        lines.append("\n## Sentiment Analysis")
        if sentiments:
            total = sum(sentiments.values()) or 1
            for k, v in sentiments.items():
                lines.append(f"- {k.title()}: {v} segments ({(v/total)*100:.1f}%)")
        else:
            lines.append("_No sentiment analysis available._")

        return "\n".join(lines) + "\n"

    def _render_full_transcript(self, result: TranscriptionResult) -> str:
        lines: list[str] = ["# Full Transcript with Speaker Timestamps", ""]

        # If utterances exist, format with timestamps; else fall back to raw transcript
        if result.utterances:
            # Optional: section headers every ~10 minutes using timestamp fence
            last_section_min = -999
            for utt in result.utterances:
                current_min = int(utt.start // 600)
                if current_min > last_section_min:
                    # New section header
                    lines.append(f"\n## Section starting at [{self._format_hms(utt.start)}]\n")
                    last_section_min = current_min
                lines.append(
                    f"[{self._format_hms(utt.start)}] Speaker {utt.speaker + 1}: {utt.text}"
                )
        else:
            lines.append(result.transcript)

        # Speaker statistics (roughly by utterance duration)
        if result.utterances and result.duration > 0:
            lines.append("\n---\n\n## Speaker Statistics")
            totals: dict[int, float] = {}
            for utt in result.utterances:
                totals[utt.speaker] = totals.get(utt.speaker, 0.0) + max(0.0, utt.end - utt.start)
            for speaker_id, seconds in sorted(totals.items()):
                pct = (seconds / result.duration) * 100
                lines.append(
                    f"- Speaker {speaker_id + 1}: {self._format_hms(seconds)} ({pct:.1f}%)"
                )

        return "\n".join(lines) + "\n"

    def _render_key_insights(self, result: TranscriptionResult) -> str:
        lines: list[str] = ["# Key Insights and Actionable Takeaways", ""]

        # Simple heuristic: extract sentences containing action language
        candidates = self._find_action_sentences(result.transcript)
        if not candidates:
            lines.append("_No specific insights identified. Review transcript for highlights._")
            return "\n".join(lines) + "\n"

        lines.append("## Strategic Insights\n")
        for idx, sentence in enumerate(candidates[:15], 1):
            lines.append(f"### {idx}. Insight")
            lines.append(f'**Evidence:** "{sentence.strip()}"')
            # Try to map to a nearest utterance timestamp if available
            ts = self._approx_timestamp_for_sentence(sentence, result.utterances or [])
            if ts is not None:
                lines.append(f"**Timestamp:** [{self._format_hms(ts)}]")
            lines.append("**Implications:** Describe potential impact or meaning.")
            lines.append("**Action Items:** Define next steps or owners.")
            lines.append("")

        return "\n".join(lines) + "\n"

    # ---------------------- Helpers ----------------------
    def _format_hms(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _overall_sentiment(self, result: TranscriptionResult) -> str:
        dist = result.sentiment_distribution or {}
        if not dist:
            return "Unknown"
        return max(dist.items(), key=lambda x: x[1])[0].title()

    def _fallback_summary(self, transcript: str) -> str:
        # Very simple fallback: first ~3 sentences or 300 chars
        if not transcript:
            return "No summary available."
        parts = [s.strip() for s in transcript.split(".") if s.strip()]
        if len(parts) >= 3:
            return ". ".join(parts[:3]) + "."
        return (transcript[:300] + ("..." if len(transcript) > 300 else "")).strip()

    def _find_action_sentences(self, transcript: str) -> list[str]:
        if not transcript:
            return []
        action_keywords = [
            "should",
            "need to",
            "must",
            "will",
            "plan to",
            "going to",
            "have to",
            "recommend",
            "priority",
        ]
        sentences = [s.strip() for s in transcript.split(".") if s.strip()]
        return [s for s in sentences if any(k in s.lower() for k in action_keywords)]

    def _approx_timestamp_for_sentence(
        self, sentence: str, utterances: list[TranscriptionUtterance]
    ) -> float | None:
        if not utterances:
            return None
        # naive approach: try first utterance containing a significant substring
        key = sentence[:20].lower()
        for utt in utterances:
            if key and key in (utt.text or "").lower():
                return utt.start
        # fallback: start of the session
        return 0.0
