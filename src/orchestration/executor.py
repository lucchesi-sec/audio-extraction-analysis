"""Step execution strategies for workflow orchestration."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from .workflow import StepResult, StepStatus, WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)


class StepExecutor(ABC):
    """Abstract base class for step execution strategies."""

    @abstractmethod
    def execute(
        self, steps: List[WorkflowStep], state: WorkflowState, max_workers: int = 10
    ) -> List[StepResult]:
        """Execute workflow steps.

        Args:
            steps: Steps to execute
            state: Current workflow state
            max_workers: Maximum concurrent workers

        Returns:
            List of step results
        """
        pass


class SequentialExecutor(StepExecutor):
    """Execute steps sequentially."""

    def execute(
        self, steps: List[WorkflowStep], state: WorkflowState, max_workers: int = 10
    ) -> List[StepResult]:
        """Execute steps one by one.

        Args:
            steps: Steps to execute
            state: Workflow state
            max_workers: Ignored for sequential execution

        Returns:
            Step results
        """
        results = []

        for step in steps:
            logger.info(f"Executing step {step.name} sequentially")
            result = self._execute_step(step, state)
            results.append(result)

            if result.status == StepStatus.FAILED and not step.metadata.get("continue_on_failure"):
                logger.warning(f"Step {step.name} failed, stopping sequential execution")
                break

        return results

    def _execute_step(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute a single step.

        Args:
            step: Step to execute
            state: Workflow state

        Returns:
            Step result
        """
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)

        try:
            # Execute step function
            step_output = step.function(*step.args, **step.kwargs, context=state.context)

            # Update state context
            state.context[f"{step.id}_result"] = step_output
            state.completed_steps.add(step.id)

            result.complete(result=step_output)

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}")
            state.failed_steps.add(step.id)
            result.complete(error=e)

        state.step_results[step.id] = result
        return result


class ParallelExecutor(StepExecutor):
    """Execute steps in parallel using threads."""

    def execute(
        self, steps: List[WorkflowStep], state: WorkflowState, max_workers: int = 10
    ) -> List[StepResult]:
        """Execute steps in parallel.

        Args:
            steps: Steps to execute
            state: Workflow state
            max_workers: Maximum parallel workers

        Returns:
            Step results
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all steps
            futures = {executor.submit(self._execute_step, step, state): step for step in steps}

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                step = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Step {step.id} execution failed: {e}")
                    result = StepResult(step_id=step.id, status=StepStatus.FAILED, error=e)
                    results.append(result)

        return results

    def _execute_step(self, step: WorkflowStep, state: WorkflowState) -> StepResult:
        """Execute a single step (thread-safe).

        Args:
            step: Step to execute
            state: Workflow state

        Returns:
            Step result
        """
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)

        try:
            # Execute step function
            step_output = step.function(*step.args, **step.kwargs, context=state.context.copy())

            # Thread-safe state update
            with threading.Lock():
                state.context[f"{step.id}_result"] = step_output
                state.completed_steps.add(step.id)

            result.complete(result=step_output)

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}")
            with threading.Lock():
                state.failed_steps.add(step.id)
            result.complete(error=e)

        state.step_results[step.id] = result
        return result


class AsyncExecutor(StepExecutor):
    """Execute steps asynchronously using asyncio."""

    def execute(
        self, steps: List[WorkflowStep], state: WorkflowState, max_workers: int = 10
    ) -> List[StepResult]:
        """Execute steps asynchronously.

        Args:
            steps: Steps to execute
            state: Workflow state
            max_workers: Maximum concurrent tasks

        Returns:
            Step results
        """
        # Run async execution in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(self._execute_async(steps, state, max_workers))
            return results
        finally:
            loop.close()

    async def _execute_async(
        self, steps: List[WorkflowStep], state: WorkflowState, max_workers: int
    ) -> List[StepResult]:
        """Execute steps asynchronously.

        Args:
            steps: Steps to execute
            state: Workflow state
            max_workers: Maximum concurrent tasks

        Returns:
            Step results
        """
        semaphore = asyncio.Semaphore(max_workers)
        tasks = []

        for step in steps:
            task = asyncio.create_task(self._execute_step_async(step, state, semaphore))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                step_result = StepResult(
                    step_id=steps[i].id, status=StepStatus.FAILED, error=result
                )
            else:
                step_result = result
            processed_results.append(step_result)

        return processed_results

    async def _execute_step_async(
        self, step: WorkflowStep, state: WorkflowState, semaphore: asyncio.Semaphore
    ) -> StepResult:
        """Execute step asynchronously.

        Args:
            step: Step to execute
            state: Workflow state
            semaphore: Concurrency limiter

        Returns:
            Step result
        """
        async with semaphore:
            result = StepResult(step_id=step.id, status=StepStatus.RUNNING)

            try:
                # Check if function is async
                if asyncio.iscoroutinefunction(step.function):
                    step_output = await step.function(
                        *step.args, **step.kwargs, context=state.context
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    step_output = await loop.run_in_executor(
                        None, step.function, *step.args, **{**step.kwargs, "context": state.context}
                    )

                # Update state
                state.context[f"{step.id}_result"] = step_output
                state.completed_steps.add(step.id)

                result.complete(result=step_output)

            except Exception as e:
                logger.error(f"Step {step.id} failed: {e}")
                state.failed_steps.add(step.id)
                result.complete(error=e)

            state.step_results[step.id] = result
            return result


