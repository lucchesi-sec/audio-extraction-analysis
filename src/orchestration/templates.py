"""Pre-defined workflow templates for common operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .workflow import ExecutionMode, WorkflowDefinition

logger = logging.getLogger(__name__)


class BatchTranscriptionWorkflow:
    """Workflow for batch transcription of multiple audio files."""

    @staticmethod
    def create(
        files: List[Path],
        provider: str = "auto",
        output_dir: Optional[Path] = None,
        parallel: bool = True,
    ) -> WorkflowDefinition:
        """Create batch transcription workflow.

        Args:
            files: List of audio files to transcribe
            provider: Transcription provider
            output_dir: Output directory for transcriptions
            parallel: Process files in parallel

        Returns:
            Workflow definition
        """
        steps = []

        # Add validation step
        steps.append(
            {
                "id": "validate_files",
                "name": "Validate Input Files",
                "function": "validate_audio_files",
                "args": (files,),
                "kwargs": {"provider": provider},
                "metadata": {"critical": True},
            }
        )

        # Add transcription steps for each file
        for i, file_path in enumerate(files):
            step_id = f"transcribe_{i}"
            steps.append(
                {
                    "id": step_id,
                    "name": f"Transcribe {file_path.name}",
                    "function": "transcribe_audio",
                    "args": (file_path,),
                    "kwargs": {"provider": provider, "output_dir": output_dir},
                    "dependencies": ["validate_files"],
                    "max_retries": 3,
                    "timeout": 600,
                    "metadata": {"file_index": i, "file_path": str(file_path)},
                }
            )

        # Add aggregation step
        steps.append(
            {
                "id": "aggregate_results",
                "name": "Aggregate Transcription Results",
                "function": "aggregate_transcriptions",
                "kwargs": {"output_dir": output_dir},
                "dependencies": [f"transcribe_{i}" for i in range(len(files))],
                "metadata": {"final": True},
            }
        )

        return WorkflowDefinition(
            name="Batch Transcription",
            description=f"Transcribe {len(files)} audio files",
            steps=steps,
            execution_mode=ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL,
            max_parallel=10 if parallel else 1,
            allow_partial_success=True,
            enable_rollback=False,
        )


class AnalysisWorkflow:
    """Workflow for complete audio analysis pipeline."""

    @staticmethod
    def create(
        input_file: Path,
        provider: str = "auto",
        analysis_types: Optional[List[str]] = None,
        output_format: str = "markdown",
    ) -> WorkflowDefinition:
        """Create analysis workflow.

        Args:
            input_file: Audio file to analyze
            provider: Transcription provider
            analysis_types: Types of analysis to perform
            output_format: Output format for results

        Returns:
            Workflow definition
        """
        if analysis_types is None:
            analysis_types = ["transcription", "summary", "key_topics", "sentiment"]

        steps = []

        # Audio extraction
        steps.append(
            {
                "id": "extract_audio",
                "name": "Extract Audio",
                "function": "extract_audio",
                "args": (input_file,),
                "timeout": 300,
                "metadata": {"stage": "preprocessing"},
            }
        )

        # Transcription
        steps.append(
            {
                "id": "transcribe",
                "name": "Transcribe Audio",
                "function": "transcribe_audio",
                "kwargs": {"provider": provider},
                "dependencies": ["extract_audio"],
                "max_retries": 3,
                "timeout": 600,
                "rollback_function": "cleanup_temp_files",
                "metadata": {"stage": "transcription"},
            }
        )

        # Conditional analysis steps
        if "summary" in analysis_types:
            steps.append(
                {
                    "id": "generate_summary",
                    "name": "Generate Summary",
                    "function": "generate_summary",
                    "dependencies": ["transcribe"],
                    "timeout": 120,
                    "metadata": {"stage": "analysis", "type": "summary"},
                }
            )

        if "key_topics" in analysis_types:
            steps.append(
                {
                    "id": "extract_topics",
                    "name": "Extract Key Topics",
                    "function": "extract_key_topics",
                    "dependencies": ["transcribe"],
                    "timeout": 120,
                    "metadata": {"stage": "analysis", "type": "topics"},
                }
            )

        if "sentiment" in analysis_types:
            steps.append(
                {
                    "id": "analyze_sentiment",
                    "name": "Analyze Sentiment",
                    "function": "analyze_sentiment",
                    "dependencies": ["transcribe"],
                    "timeout": 120,
                    "metadata": {"stage": "analysis", "type": "sentiment"},
                }
            )

        if "speakers" in analysis_types:
            steps.append(
                {
                    "id": "identify_speakers",
                    "name": "Identify Speakers",
                    "function": "identify_speakers",
                    "dependencies": ["transcribe"],
                    "timeout": 180,
                    "metadata": {"stage": "analysis", "type": "speakers"},
                }
            )

        # Format output
        format_deps = ["transcribe"]
        for analysis in analysis_types:
            if analysis != "transcription":
                format_deps.append(f"{analysis.replace('_', '_')}")

        steps.append(
            {
                "id": "format_output",
                "name": f"Format Output as {output_format}",
                "function": "format_analysis_output",
                "kwargs": {"format": output_format},
                "dependencies": format_deps,
                "metadata": {"stage": "output", "format": output_format},
            }
        )

        return WorkflowDefinition(
            name="Audio Analysis Pipeline",
            description=f"Complete analysis of {input_file.name}",
            steps=steps,
            execution_mode=ExecutionMode.PARALLEL,
            max_parallel=5,
            timeout=1800,  # 30 minutes total
            allow_partial_success=True,
            enable_rollback=True,
        )


class RetryWorkflow:
    """Workflow with automatic fallback to alternative providers."""

    @staticmethod
    def create(input_file: Path, providers: List[str], max_attempts: int = 3) -> WorkflowDefinition:
        """Create retry workflow with provider fallback.

        Args:
            input_file: Audio file to process
            providers: List of providers in priority order
            max_attempts: Maximum retry attempts per provider

        Returns:
            Workflow definition
        """
        steps = []

        # Validate providers
        steps.append(
            {
                "id": "validate_providers",
                "name": "Validate Available Providers",
                "function": "validate_providers",
                "args": (providers,),
                "metadata": {"critical": True},
            }
        )

        # Create cascading provider steps
        for i, provider in enumerate(providers):
            step_id = f"transcribe_provider_{i}"

            # Dependencies include previous failed attempts
            deps = ["validate_providers"]
            if i > 0:
                deps.append(
                    {
                        "step_id": f"transcribe_provider_{i-1}",
                        "condition": lambda result: result is None
                        or result.get("status") == "failed",
                        "required": False,
                    }
                )

            steps.append(
                {
                    "id": step_id,
                    "name": f"Transcribe with {provider}",
                    "function": "transcribe_with_provider",
                    "args": (input_file,),
                    "kwargs": {"provider": provider},
                    "dependencies": deps,
                    "max_retries": max_attempts,
                    "timeout": 600,
                    "error_handler": "log_provider_error",
                    "metadata": {"provider": provider, "priority": i, "fallback": i > 0},
                }
            )

        # Select best result
        steps.append(
            {
                "id": "select_result",
                "name": "Select Best Transcription",
                "function": "select_best_transcription",
                "dependencies": [f"transcribe_provider_{i}" for i in range(len(providers))],
                "metadata": {"final": True},
            }
        )

        return WorkflowDefinition(
            name="Retry with Fallback",
            description=f"Transcribe with fallback across {len(providers)} providers",
            steps=steps,
            execution_mode=ExecutionMode.SEQUENTIAL,
            allow_partial_success=True,
            enable_rollback=False,
        )


class QualityCheckWorkflow:
    """Workflow with quality validation and correction."""

    @staticmethod
    def create(input_file: Path, quality_threshold: float = 0.8) -> WorkflowDefinition:
        """Create quality check workflow.

        Args:
            input_file: Audio file to process
            quality_threshold: Minimum quality score

        Returns:
            Workflow definition
        """
        steps = [
            {
                "id": "initial_transcription",
                "name": "Initial Transcription",
                "function": "transcribe_audio",
                "args": (input_file,),
                "metadata": {"attempt": 1},
            },
            {
                "id": "quality_check",
                "name": "Check Transcription Quality",
                "function": "check_transcription_quality",
                "kwargs": {"threshold": quality_threshold},
                "dependencies": ["initial_transcription"],
                "metadata": {"validation": True},
            },
            {
                "id": "enhance_audio",
                "name": "Enhance Audio Quality",
                "function": "enhance_audio",
                "args": (input_file,),
                "dependencies": [
                    {
                        "step_id": "quality_check",
                        "condition": lambda result: result.get("quality_score", 0)
                        < quality_threshold,
                    }
                ],
                "metadata": {"conditional": True},
            },
            {
                "id": "retry_transcription",
                "name": "Retry with Enhanced Audio",
                "function": "transcribe_audio",
                "dependencies": ["enhance_audio"],
                "kwargs": {"enhanced": True},
                "metadata": {"attempt": 2},
            },
            {
                "id": "final_quality_check",
                "name": "Final Quality Check",
                "function": "check_transcription_quality",
                "kwargs": {"threshold": quality_threshold},
                "dependencies": ["retry_transcription"],
                "metadata": {"validation": True, "final": True},
            },
            {
                "id": "post_process",
                "name": "Post-process Transcription",
                "function": "post_process_transcription",
                "dependencies": ["final_quality_check"],
                "metadata": {"stage": "finalization"},
            },
        ]

        return WorkflowDefinition(
            name="Quality Check Pipeline",
            description="Transcription with quality validation and enhancement",
            steps=steps,
            execution_mode=ExecutionMode.SEQUENTIAL,
            allow_partial_success=False,
            enable_rollback=True,
        )


def get_workflow_template(template_name: str, **kwargs) -> Optional[WorkflowDefinition]:
    """Get pre-defined workflow template.

    Args:
        template_name: Name of template
        **kwargs: Template-specific parameters

    Returns:
        Workflow definition or None
    """
    templates = {
        "batch": BatchTranscriptionWorkflow,
        "analysis": AnalysisWorkflow,
        "retry": RetryWorkflow,
        "quality": QualityCheckWorkflow,
    }

    template_class = templates.get(template_name.lower())
    if template_class:
        return template_class.create(**kwargs)

    logger.warning(f"Unknown workflow template: {template_name}")
    return None
