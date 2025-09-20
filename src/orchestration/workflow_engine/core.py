"""
Core DAG orchestration engine.

This module contains the main workflow orchestrator that manages DAG execution,
state management, callbacks, and metrics collection.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx

from .executors import SequentialExecutor, StepExecutor
from .steps import (
    ExecutionMode,
    StepDependency,
    StepResult,
    StepStatus,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Main workflow orchestration engine."""

    def __init__(
        self,
        state_manager: Optional["WorkflowStateManager"] = None,
        executor: Optional[StepExecutor] = None,
        enable_metrics: bool = True,
    ):
        """Initialize workflow orchestrator.

        Args:
            state_manager: State persistence manager
            executor: Step execution strategy
            enable_metrics: Enable metrics collection
        """
        # Import here to avoid circular imports
        from ..state_manager import InMemoryStateManager

        self.state_manager = state_manager or InMemoryStateManager()
        self.executor = executor or SequentialExecutor()
        self.enable_metrics = enable_metrics

        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._states: Dict[str, WorkflowState] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = Lock()

        # Metrics
        self._metrics = {
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "steps_executed": 0,
            "steps_failed": 0,
            "total_duration": 0.0,
        }

    def register_workflow(self, definition: WorkflowDefinition) -> str:
        """Register a workflow definition.

        Args:
            definition: Workflow definition

        Returns:
            Workflow ID
        """
        with self._lock:
            self._workflows[definition.id] = definition
            logger.info(f"Registered workflow: {definition.name} ({definition.id})")
            return definition.id

    def execute(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        async_execution: bool = False,
    ) -> Union[WorkflowResult, asyncio.Task]:
        """Execute a workflow.

        Args:
            workflow_id: Workflow ID to execute
            context: Initial workflow context
            async_execution: Execute asynchronously

        Returns:
            WorkflowResult or async task

        Raises:
            ValueError: If workflow not found
        """
        if workflow_id not in self._workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Initialize state
        state = WorkflowState(
            workflow_id=workflow_id, context=context or {}, start_time=datetime.now()
        )

        with self._lock:
            self._states[workflow_id] = state
            if self.enable_metrics:
                self._metrics["workflows_started"] += 1

        if async_execution:
            return asyncio.create_task(self._execute_async(workflow_id, state))
        else:
            return self._execute_sync(workflow_id, state)

    def _execute_sync(self, workflow_id: str, state: WorkflowState) -> WorkflowResult:
        """Execute workflow synchronously.

        Args:
            workflow_id: Workflow ID
            state: Workflow state

        Returns:
            WorkflowResult
        """
        definition = self._workflows[workflow_id]
        dag = definition.to_dag()

        try:
            state.status = WorkflowStatus.RUNNING
            self._notify_callbacks(workflow_id, "started", state)

            # Execute steps in topological order
            for step_id in nx.topological_sort(dag):
                if state.status in {WorkflowStatus.CANCELLED, WorkflowStatus.FAILED}:
                    break

                step_config = dag.nodes[step_id]
                step = self._create_step(step_id, step_config)

                # Check dependencies
                if not self.executor.check_dependencies(step, state):
                    state.skipped_steps.add(step_id)
                    continue

                # Check condition
                if step.condition and not step.condition(state.context):
                    state.skipped_steps.add(step_id)
                    logger.info(f"Skipping step {step_id} due to condition")
                    continue

                # Execute step
                state.current_step = step_id
                result = self.executor.execute_step(step, state)

                if result.status == StepStatus.FAILED:
                    if not self._handle_step_failure(step, result, state, definition):
                        state.status = WorkflowStatus.FAILED
                        break

            # Finalize workflow
            if state.status == WorkflowStatus.RUNNING:
                state.status = WorkflowStatus.COMPLETED

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            state.status = WorkflowStatus.FAILED
            state.error = e
        finally:
            state.end_time = datetime.now()
            self._finalize_workflow(workflow_id, state)

        return self._create_result(workflow_id, state)

    async def _execute_async(self, workflow_id: str, state: WorkflowState) -> WorkflowResult:
        """Execute workflow asynchronously.

        Args:
            workflow_id: Workflow ID
            state: Workflow state

        Returns:
            WorkflowResult
        """
        definition = self._workflows[workflow_id]
        dag = definition.to_dag()

        try:
            state.status = WorkflowStatus.RUNNING
            self._notify_callbacks(workflow_id, "started", state)

            # Group steps by level for parallel execution
            levels = self._get_execution_levels(dag)

            for level in levels:
                if state.status in {WorkflowStatus.CANCELLED, WorkflowStatus.FAILED}:
                    break

                # Execute steps in parallel within each level
                tasks = []
                for step_id in level:
                    step_config = dag.nodes[step_id]
                    step = self._create_step(step_id, step_config)

                    # Check dependencies
                    if not self.executor.check_dependencies(step, state):
                        state.skipped_steps.add(step_id)
                        continue

                    # Check condition
                    if step.condition and not step.condition(state.context):
                        state.skipped_steps.add(step_id)
                        continue

                    # Create async task
                    task = asyncio.create_task(self.executor.execute_step_async(step, state))
                    tasks.append((step_id, task))

                # Wait for all tasks in level to complete
                for step_id, task in tasks:
                    result = await task
                    if result.status == StepStatus.FAILED:
                        step = next(
                            s
                            for s in [self._create_step(sid, dag.nodes[sid]) for sid in level]
                            if s.id == step_id
                        )
                        if not self._handle_step_failure(step, result, state, definition):
                            state.status = WorkflowStatus.FAILED
                            break

            # Finalize workflow
            if state.status == WorkflowStatus.RUNNING:
                state.status = WorkflowStatus.COMPLETED

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            state.status = WorkflowStatus.FAILED
            state.error = e
        finally:
            state.end_time = datetime.now()
            self._finalize_workflow(workflow_id, state)

        return self._create_result(workflow_id, state)

    def _create_step(self, step_id: str, config: Dict[str, Any]) -> WorkflowStep:
        """Create workflow step from configuration.

        Args:
            step_id: Step ID
            config: Step configuration

        Returns:
            WorkflowStep instance
        """
        # Get function from config (would be resolved from registry in production)
        function = config.get("function", lambda: None)

        return WorkflowStep(
            id=step_id,
            name=config.get("name", step_id),
            function=function,
            args=config.get("args", ()),
            kwargs=config.get("kwargs", {}),
            dependencies=[
                StepDependency(**dep) if isinstance(dep, dict) else StepDependency(dep)
                for dep in config.get("dependencies", [])
            ],
            execution_mode=ExecutionMode(config.get("execution_mode", "sequential")),
            timeout=config.get("timeout"),
            max_retries=config.get("max_retries", 3),
            rollback_function=config.get("rollback_function"),
            error_handler=config.get("error_handler"),
            condition=config.get("condition"),
            metadata=config.get("metadata", {}),
        )

    def _handle_step_failure(
        self,
        step: WorkflowStep,
        result: StepResult,
        state: WorkflowState,
        definition: WorkflowDefinition,
    ) -> bool:
        """Handle step failure with retry and rollback.

        Args:
            step: Failed step
            result: Step result
            state: Workflow state
            definition: Workflow definition

        Returns:
            True if handled, False if workflow should fail
        """
        # Try retry
        if result.retry_count < step.max_retries:
            logger.info(f"Retrying step {step.id} (attempt {result.retry_count + 1})")
            result.retry_count += 1
            retry_result = self.executor.execute_step(step, state)

            if retry_result.status == StepStatus.COMPLETED:
                return True

        # Check if partial success allowed
        if definition.allow_partial_success:
            logger.warning(f"Step {step.id} failed but continuing (partial success allowed)")
            return True

        # Try rollback if enabled
        if definition.enable_rollback and step.rollback_function:
            try:
                logger.info(f"Rolling back step {step.id}")
                step.rollback_function(state.context)
                result.status = StepStatus.ROLLED_BACK
            except Exception as e:
                logger.error(f"Rollback failed for {step.id}: {e}")

        return False

    def _get_execution_levels(self, dag: nx.DiGraph) -> List[List[str]]:
        """Get execution levels for parallel execution.

        Args:
            dag: Workflow DAG

        Returns:
            List of step IDs grouped by execution level
        """
        levels = []
        remaining = set(dag.nodes())
        completed = set()

        while remaining:
            # Find nodes with satisfied dependencies
            level = []
            for node in remaining:
                predecessors = set(dag.predecessors(node))
                if predecessors.issubset(completed):
                    level.append(node)

            if not level:
                raise ValueError("Cannot determine execution levels - possible cycle")

            levels.append(level)
            completed.update(level)
            remaining.difference_update(level)

        return levels

    def _finalize_workflow(self, workflow_id: str, state: WorkflowState):
        """Finalize workflow execution.

        Args:
            workflow_id: Workflow ID
            state: Final workflow state
        """
        # Update metrics
        if self.enable_metrics:
            if state.status == WorkflowStatus.COMPLETED:
                self._metrics["workflows_completed"] += 1
            else:
                self._metrics["workflows_failed"] += 1

            if state.get_duration():
                self._metrics["total_duration"] += state.get_duration()

        # Persist state
        self.state_manager.save_state(workflow_id, state)

        # Notify callbacks
        self._notify_callbacks(workflow_id, "completed", state)

        logger.info(
            f"Workflow {workflow_id} finished with status {state.status.value} "
            f"in {state.get_duration():.2f}s"
        )

    def _create_result(self, workflow_id: str, state: WorkflowState) -> WorkflowResult:
        """Create workflow result from final state.

        Args:
            workflow_id: Workflow ID
            state: Final state

        Returns:
            WorkflowResult
        """
        return WorkflowResult(
            workflow_id=workflow_id,
            status=state.status,
            results={
                step_id: result.result
                for step_id, result in state.step_results.items()
                if result.status == StepStatus.COMPLETED
            },
            errors=[result.error for result in state.step_results.values() if result.error],
            duration=state.get_duration() or 0.0,
            completed_steps=len(state.completed_steps),
            failed_steps=len(state.failed_steps),
            skipped_steps=len(state.skipped_steps),
            metadata=state.metadata,
        )

    def pause_workflow(self, workflow_id: str):
        """Pause a running workflow.

        Args:
            workflow_id: Workflow to pause
        """
        if workflow_id in self._states:
            self._states[workflow_id].status = WorkflowStatus.PAUSED
            logger.info(f"Paused workflow {workflow_id}")

    def resume_workflow(self, workflow_id: str):
        """Resume a paused workflow.

        Args:
            workflow_id: Workflow to resume
        """
        if workflow_id in self._states:
            state = self._states[workflow_id]
            if state.status == WorkflowStatus.PAUSED:
                state.status = WorkflowStatus.RUNNING
                logger.info(f"Resumed workflow {workflow_id}")

    def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow.

        Args:
            workflow_id: Workflow to cancel
        """
        if workflow_id in self._states:
            self._states[workflow_id].status = WorkflowStatus.CANCELLED
            logger.info(f"Cancelled workflow {workflow_id}")

    def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current workflow state.

        Args:
            workflow_id: Workflow ID

        Returns:
            Current state or None
        """
        return self._states.get(workflow_id) or self.state_manager.load_state(workflow_id)

    def add_callback(self, workflow_id: str, event: str, callback: Callable):
        """Add event callback for workflow.

        Args:
            workflow_id: Workflow ID
            event: Event name (started, completed, failed, etc.)
            callback: Callback function
        """
        key = f"{workflow_id}:{event}"
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def _notify_callbacks(self, workflow_id: str, event: str, state: WorkflowState):
        """Notify registered callbacks.

        Args:
            workflow_id: Workflow ID
            event: Event name
            state: Current state
        """
        key = f"{workflow_id}:{event}"
        for callback in self._callbacks.get(key, []):
            try:
                callback(workflow_id, event, state)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics.

        Returns:
            Metrics dictionary
        """
        return self._metrics.copy()
