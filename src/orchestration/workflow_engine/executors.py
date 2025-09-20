"""
Step execution strategies and rollback capabilities.

This module contains the execution logic for workflow steps, including 
synchronous, asynchronous, and parallel execution modes with rollback support.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from .steps import StepResult, StepStatus, WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)


class StepExecutor(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    def execute_step(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute a workflow step.
        
        Args:
            step: Step to execute
            state: Current workflow state
            
        Returns:
            StepResult
        """
        pass

    @abstractmethod
    async def execute_step_async(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute step asynchronously.
        
        Args:
            step: Step to execute
            state: Workflow state
            
        Returns:
            StepResult
        """
        pass

    def check_dependencies(self, step: WorkflowStep, state: WorkflowState) -> bool:
        """Check if step dependencies are satisfied.

        Args:
            step: Step to check
            state: Current workflow state

        Returns:
            True if dependencies are satisfied
        """
        for dep in step.dependencies:
            # Check if dependency completed
            if dep.step_id not in state.completed_steps:
                if dep.required:
                    logger.warning(f"Required dependency {dep.step_id} not completed for {step.id}")
                    return False
                continue

            # Check dependency condition
            if dep.condition:
                dep_result = state.step_results.get(dep.step_id)
                if dep_result and not dep.condition(dep_result.result):
                    logger.warning(f"Dependency condition not met for {step.id}")
                    return False

        return True


class SequentialExecutor(StepExecutor):
    """Sequential step executor with timeout and error handling."""

    def __init__(self, enable_metrics: bool = True):
        """Initialize sequential executor.
        
        Args:
            enable_metrics: Enable metrics collection
        """
        self.enable_metrics = enable_metrics
        self._metrics = {
            "steps_executed": 0,
            "steps_failed": 0,
        }

    def execute_step(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute a single workflow step.

        Args:
            step: Step to execute
            state: Current workflow state

        Returns:
            StepResult
        """
        logger.info(f"Executing step: {step.name}")
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)

        try:
            # Execute with timeout if specified
            if step.timeout:
                import signal

                def timeout_handler(_signum, _frame):
                    raise TimeoutError(f"Step {step.id} timed out after {step.timeout}s")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(step.timeout))

            # Execute step function
            step_result = step.function(*step.args, **step.kwargs, context=state.context)

            if step.timeout:
                signal.alarm(0)

            # Update context with result
            state.context[f"{step.id}_result"] = step_result
            result.complete(result=step_result)

            # Mark as completed
            state.completed_steps.add(step.id)

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}")
            result.complete(error=e)
            state.failed_steps.add(step.id)

            # Try error handler
            if step.error_handler:
                try:
                    step.error_handler(e, state.context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")

        finally:
            state.step_results[step.id] = result
            if self.enable_metrics:
                self._metrics["steps_executed"] += 1
                if result.status == StepStatus.FAILED:
                    self._metrics["steps_failed"] += 1

        return result

    async def execute_step_async(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute step asynchronously.

        Args:
            step: Step to execute
            state: Workflow state

        Returns:
            StepResult
        """
        # Wrap synchronous execution in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_step, step, state)

    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics.
        
        Returns:
            Metrics dictionary
        """
        return self._metrics.copy()
