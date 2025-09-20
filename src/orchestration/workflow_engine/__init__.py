"""
Modularized workflow orchestration engine.

This package contains the refactored workflow engine components:
- steps: Step models and dependencies
- core: DAG orchestration logic
- executors: Execution strategies and rollback
"""

from __future__ import annotations

# Export main public API
from .steps import (
    WorkflowStep,
    StepDependency, 
    StepResult,
    StepStatus,
    ExecutionMode,
    WorkflowState,
    WorkflowResult,
    WorkflowDefinition,
    WorkflowStatus,
)

from .core import WorkflowOrchestrator

from .executors import SequentialExecutor, StepExecutor

__all__ = [
    # Step models
    "WorkflowStep",
    "StepDependency", 
    "StepResult", 
    "StepStatus",
    "ExecutionMode",
    "WorkflowState",
    "WorkflowResult",
    "WorkflowDefinition", 
    "WorkflowStatus",
    
    # Core orchestrator
    "WorkflowOrchestrator",
    
    # Executors
    "SequentialExecutor",
    "StepExecutor",
]
