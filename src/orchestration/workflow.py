"""Core workflow orchestration implementation (compatibility shim).

This module is now a thin compatibility layer that re-exports the refactored
workflow engine from src/orchestration/workflow_engine/.

The original monolithic workflow.py (770 lines) has been split into:
- workflow_engine/steps.py (150 LOC) - Data models and step definitions  
- workflow_engine/executors.py (140 LOC) - Execution strategies and rollback
- workflow_engine/core.py (270 LOC) - Main DAG orchestration logic
- workflow.py (shim, <50 LOC) - Backward compatibility layer

This maintains the public API while improving maintainability.
"""

from __future__ import annotations

# Re-export public API from the new modular engine
from .workflow_engine import (
    # Core orchestrator
    WorkflowOrchestrator,
    
    # Data models  
    WorkflowDefinition,
    WorkflowStep,
    StepDependency,
    StepResult,
    WorkflowState,
    WorkflowResult,
    
    # Enums
    StepStatus,
    ExecutionMode, 
    WorkflowStatus,
    
    # Executors
    SequentialExecutor,
    StepExecutor,
)

# Backward compatibility imports
from .state_manager import InMemoryStateManager, WorkflowStateManager

__all__ = [
    # Main classes
    "WorkflowOrchestrator",
    "WorkflowDefinition", 
    "WorkflowStep",
    "StepDependency",
    "StepResult",
    "WorkflowState", 
    "WorkflowResult",
    
    # Status enums
    "StepStatus",
    "ExecutionMode",
    "WorkflowStatus", 
    
    # Executors
    "SequentialExecutor", 
    "StepExecutor",
    
    # Legacy compatibility
    "InMemoryStateManager",
    "WorkflowStateManager",
]

