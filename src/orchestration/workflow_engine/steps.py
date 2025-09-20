"""
Workflow step models and data structures.

This module defines the core data models for workflow steps, dependencies,
results, and state management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import networkx as nx
from pydantic import BaseModel, Field, field_validator


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class StepStatus(Enum):
    """Individual step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"
    ROLLED_BACK = "rolled_back"


class ExecutionMode(Enum):
    """Step execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    CONDITIONAL = "conditional"


@dataclass
class StepDependency:
    """Dependency definition for workflow steps."""

    step_id: str
    condition: Optional[Callable[[Any], bool]] = None
    required: bool = True
    wait_for_completion: bool = True


@dataclass
class WorkflowStep:
    """Individual workflow step definition."""

    id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    dependencies: List[StepDependency] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    rollback_function: Optional[Callable] = None
    error_handler: Optional[Callable] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make step hashable for use in sets."""
        return hash(self.id)


@dataclass
class StepResult:
    """Result from executing a workflow step."""

    step_id: str
    status: StepStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, result: Any = None, error: Optional[Exception] = None):
        """Mark step as complete."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        if error:
            self.status = StepStatus.FAILED
            self.error = error
        else:
            self.status = StepStatus.COMPLETED
            self.result = result


@dataclass
class WorkflowState:
    """Current state of workflow execution."""

    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if workflow is in terminal state."""
        return self.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }

    def get_duration(self) -> Optional[float]:
        """Get workflow execution duration."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@dataclass
class WorkflowResult:
    """Final result of workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    results: Dict[str, Any]
    errors: List[Exception]
    duration: float
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    metadata: Dict[str, Any]


class WorkflowDefinition(BaseModel):
    """Workflow definition with validation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    steps: List[Dict[str, Any]]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel: int = Field(10, ge=1, le=100)
    timeout: Optional[float] = None
    allow_partial_success: bool = False
    enable_rollback: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v):
        """Validate step definitions."""
        if not v:
            raise ValueError("Workflow must have at least one step")

        step_ids = set()
        for step in v:
            if "id" not in step:
                raise ValueError("Each step must have an 'id'")
            if step["id"] in step_ids:
                raise ValueError(f"Duplicate step ID: {step['id']}")
            step_ids.add(step["id"])

        return v

    def to_dag(self) -> nx.DiGraph:
        """Convert workflow definition to directed acyclic graph."""
        dag = nx.DiGraph()

        # Add nodes
        for step in self.steps:
            dag.add_node(step["id"], **step)

        # Add edges based on dependencies
        for step in self.steps:
            if "dependencies" in step:
                for dep in step["dependencies"]:
                    if isinstance(dep, str):
                        dag.add_edge(dep, step["id"])
                    elif isinstance(dep, dict):
                        dag.add_edge(dep["step_id"], step["id"], **dep)

        # Validate DAG
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Workflow contains cycles")

        return dag
