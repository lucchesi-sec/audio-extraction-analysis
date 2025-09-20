"""Workflow state management implementations."""
from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from .workflow import StepResult, StepStatus, WorkflowState, WorkflowStatus

logger = logging.getLogger(__name__)


class WorkflowStateManager(ABC):
    """Abstract base class for workflow state persistence."""

    @abstractmethod
    def save_state(self, workflow_id: str, state: WorkflowState) -> bool:
        """Save workflow state.

        Args:
            workflow_id: Workflow identifier
            state: State to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Saved state or None
        """
        pass

    @abstractmethod
    def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def list_states(self) -> Dict[str, WorkflowStatus]:
        """List all saved workflow states.

        Returns:
            Dictionary of workflow ID to status
        """
        pass

    @abstractmethod
    def cleanup_old_states(self, days: int = 30) -> int:
        """Clean up old completed/failed states.

        Args:
            days: Age threshold in days

        Returns:
            Number of states cleaned up
        """
        pass


class InMemoryStateManager(WorkflowStateManager):
    """In-memory state manager for development/testing."""

    def __init__(self):
        """Initialize in-memory state manager."""
        self._states: Dict[str, WorkflowState] = {}
        self._lock = Lock()

    def save_state(self, workflow_id: str, state: WorkflowState) -> bool:
        """Save state in memory.

        Args:
            workflow_id: Workflow ID
            state: State to save

        Returns:
            True if successful
        """
        with self._lock:
            self._states[workflow_id] = state
            logger.debug(f"Saved state for workflow {workflow_id} in memory")
            return True

    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load state from memory.

        Args:
            workflow_id: Workflow ID

        Returns:
            State or None
        """
        with self._lock:
            return self._states.get(workflow_id)

    def delete_state(self, workflow_id: str) -> bool:
        """Delete state from memory.

        Args:
            workflow_id: Workflow ID

        Returns:
            True if deleted
        """
        with self._lock:
            if workflow_id in self._states:
                del self._states[workflow_id]
                logger.debug(f"Deleted state for workflow {workflow_id}")
                return True
            return False

    def list_states(self) -> Dict[str, WorkflowStatus]:
        """List all workflow states.

        Returns:
            Dictionary of workflow ID to status
        """
        with self._lock:
            return {wf_id: state.status for wf_id, state in self._states.items()}

    def cleanup_old_states(self, days: int = 30) -> int:
        """Clean up old states (no-op for in-memory).

        Args:
            days: Age threshold

        Returns:
            Number cleaned (always 0)
        """
        # In-memory states don't persist, so no cleanup needed
        return 0


class PersistentStateManager(WorkflowStateManager):
    """Persistent state manager using SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize persistent state manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path.home() / ".audio_extraction" / "workflows.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Create workflows table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create state table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_states (
                    workflow_id TEXT PRIMARY KEY,
                    state_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflows_status 
                ON workflows(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflows_created 
                ON workflows(created_at)
            """
            )

            conn.commit()
            logger.info(f"Initialized workflow database at {self.db_path}")

    def save_state(self, workflow_id: str, state: WorkflowState) -> bool:
        """Save state to database.

        Args:
            workflow_id: Workflow ID
            state: State to save

        Returns:
            True if successful
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    # Serialize state
                    state_data = self._serialize_state(state)

                    # Update or insert workflow record
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO workflows 
                        (workflow_id, status, current_step, start_time, end_time, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (
                            workflow_id,
                            state.status.value,
                            state.current_step,
                            state.start_time.isoformat() if state.start_time else None,
                            state.end_time.isoformat() if state.end_time else None,
                        ),
                    )

                    # Update or insert state data
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO workflow_states 
                        (workflow_id, state_data, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                        (workflow_id, state_data),
                    )

                    conn.commit()
                    logger.debug(f"Persisted state for workflow {workflow_id}")
                    return True

        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
            return False

    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load state from database.

        Args:
            workflow_id: Workflow ID

        Returns:
            State or None
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        SELECT state_data FROM workflow_states
                        WHERE workflow_id = ?
                    """,
                        (workflow_id,),
                    )

                    row = cursor.fetchone()
                    if row:
                        return self._deserialize_state(row[0])

                    return None

        except Exception as e:
            logger.error(f"Failed to load workflow state: {e}")
            return None

    def delete_state(self, workflow_id: str) -> bool:
        """Delete state from database.

        Args:
            workflow_id: Workflow ID

        Returns:
            True if deleted
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    # Delete from both tables
                    cursor.execute(
                        "DELETE FROM workflow_states WHERE workflow_id = ?", (workflow_id,)
                    )
                    cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))

                    conn.commit()

                    if cursor.rowcount > 0:
                        logger.debug(f"Deleted state for workflow {workflow_id}")
                        return True

                    return False

        except Exception as e:
            logger.error(f"Failed to delete workflow state: {e}")
            return False

    def list_states(self) -> Dict[str, WorkflowStatus]:
        """List all workflow states.

        Returns:
            Dictionary of workflow ID to status
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    cursor.execute("SELECT workflow_id, status FROM workflows")

                    return {row[0]: WorkflowStatus(row[1]) for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to list workflow states: {e}")
            return {}

    def cleanup_old_states(self, days: int = 30) -> int:
        """Clean up old completed/failed states.

        Args:
            days: Age threshold in days

        Returns:
            Number of states cleaned up
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()

                    # Delete old completed/failed workflows
                    cursor.execute(
                        """
                        DELETE FROM workflow_states 
                        WHERE workflow_id IN (
                            SELECT workflow_id FROM workflows
                            WHERE status IN ('completed', 'failed', 'cancelled')
                            AND updated_at < datetime('now', '-' || ? || ' days')
                        )
                    """,
                        (days,),
                    )

                    states_deleted = cursor.rowcount

                    cursor.execute(
                        """
                        DELETE FROM workflows
                        WHERE status IN ('completed', 'failed', 'cancelled')
                        AND updated_at < datetime('now', '-' || ? || ' days')
                    """,
                        (days,),
                    )

                    conn.commit()

                    logger.info(f"Cleaned up {states_deleted} old workflow states")
                    return states_deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0

    def _serialize_state(self, state: WorkflowState) -> bytes:
        """Serialize workflow state for storage.

        Args:
            state: State to serialize

        Returns:
            Serialized state bytes
        """
        # Convert to dictionary for JSON serialization
        state_dict = {
            "workflow_id": state.workflow_id,
            "status": state.status.value,
            "current_step": state.current_step,
            "completed_steps": list(state.completed_steps),
            "failed_steps": list(state.failed_steps),
            "skipped_steps": list(state.skipped_steps),
            "step_results": {
                step_id: {
                    "step_id": result.step_id,
                    "status": result.status.value,
                    "result": str(result.result),  # Convert to string for safety
                    "error": str(result.error) if result.error else None,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "duration": result.duration,
                    "retry_count": result.retry_count,
                    "metadata": result.metadata,
                }
                for step_id, result in state.step_results.items()
            },
            "context": state.context,
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "end_time": state.end_time.isoformat() if state.end_time else None,
            "error": str(state.error) if state.error else None,
            "metadata": state.metadata,
        }

        return json.dumps(state_dict).encode("utf-8")

    def _deserialize_state(self, data: bytes) -> WorkflowState:
        """Deserialize workflow state from storage.

        Args:
            data: Serialized state bytes

        Returns:
            WorkflowState instance
        """
        state_dict = json.loads(data.decode("utf-8"))

        # Reconstruct state
        state = WorkflowState(
            workflow_id=state_dict["workflow_id"],
            status=WorkflowStatus(state_dict["status"]),
            current_step=state_dict.get("current_step"),
            completed_steps=set(state_dict.get("completed_steps", [])),
            failed_steps=set(state_dict.get("failed_steps", [])),
            skipped_steps=set(state_dict.get("skipped_steps", [])),
            context=state_dict.get("context", {}),
            metadata=state_dict.get("metadata", {}),
        )

        # Reconstruct step results
        for step_id, result_dict in state_dict.get("step_results", {}).items():
            result = StepResult(
                step_id=result_dict["step_id"],
                status=StepStatus(result_dict["status"]),
                result=result_dict.get("result"),
                error=Exception(result_dict["error"]) if result_dict.get("error") else None,
                start_time=datetime.fromisoformat(result_dict["start_time"]),
                end_time=(
                    datetime.fromisoformat(result_dict["end_time"])
                    if result_dict.get("end_time")
                    else None
                ),
                duration=result_dict.get("duration"),
                retry_count=result_dict.get("retry_count", 0),
                metadata=result_dict.get("metadata", {}),
            )
            state.step_results[step_id] = result

        # Reconstruct timestamps
        if state_dict.get("start_time"):
            state.start_time = datetime.fromisoformat(state_dict["start_time"])
        if state_dict.get("end_time"):
            state.end_time = datetime.fromisoformat(state_dict["end_time"])

        if state_dict.get("error"):
            state.error = Exception(state_dict["error"])

        return state


