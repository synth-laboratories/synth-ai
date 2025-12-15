"""Abstract base class for trace storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..abstractions import SessionTrace


class TraceStorage(ABC):
    """Abstract base class for trace storage implementations."""

    @abstractmethod
    async def initialize(self):
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def insert_session_trace(self, trace: SessionTrace) -> str:
        """Insert a complete session trace.

        Args:
            trace: The session trace to insert

        Returns:
            The session ID
        """
        pass

    @abstractmethod
    async def get_session_trace(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session trace by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session trace data or None if not found
        """
        pass

    @abstractmethod
    async def query_traces(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a query and return results.

        Args:
            query: The SQL query to execute
            params: Optional query parameters

        Returns:
            Query results as a DataFrame-like object or list of dict records
        """
        pass

    @abstractmethod
    async def get_model_usage(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_name: str | None = None,
    ) -> Any:
        """Get model usage statistics.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            model_name: Optional model name filter

        Returns:
            Model usage statistics as a DataFrame-like object or list of dict records
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def close(self):
        """Close the storage connection."""
        pass

    # Incremental helpers -------------------------------------------------

    @abstractmethod
    async def ensure_session(
        self,
        session_id: str,
        *,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Ensure a session row exists for the given session id."""
        pass

    @abstractmethod
    async def ensure_timestep(
        self,
        session_id: str,
        *,
        step_id: str,
        step_index: int,
        turn_number: int | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ensure a timestep row exists and return its database id."""
        pass

    @abstractmethod
    async def insert_event_row(
        self,
        session_id: str,
        *,
        timestep_db_id: int | None,
        event: Any,
        metadata_override: dict[str, Any] | None = None,
    ) -> int:
        """Insert an event and return its database id."""
        pass

    @abstractmethod
    async def insert_message_row(
        self,
        session_id: str,
        *,
        timestep_db_id: int | None,
        message_type: str,
        content: Any,
        event_time: float | None = None,
        message_time: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a message row linked to a session/timestep."""
        pass

    @abstractmethod
    async def insert_outcome_reward(
        self,
        session_id: str,
        *,
        total_reward: float,
        achievements_count: int,
        total_steps: int,
        reward_metadata: dict | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> int:
        """Record an outcome reward for a session."""
        pass

    @abstractmethod
    async def insert_event_reward(
        self,
        session_id: str,
        *,
        event_id: int,
        message_id: int | None = None,
        turn_number: int | None = None,
        reward_value: float = 0.0,
        reward_type: str | None = None,
        key: str | None = None,
        annotation: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> int:
        """Record a reward tied to a specific event."""
        pass

    # Optional experiment management methods
    async def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> str:
        """Create a new experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    async def link_session_to_experiment(self, session_id: str, experiment_id: str):
        """Link a session to an experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    async def get_sessions_by_experiment(
        self, experiment_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get all sessions for an experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    # Batch operations
    async def batch_insert_sessions(
        self, traces: list[SessionTrace], batch_size: int | None = 1000
    ) -> list[str]:
        """Batch insert multiple session traces.

        Default implementation calls insert_session_trace for each trace.
        Subclasses can override for more efficient batch operations.
        """
        inserted_ids = []
        for trace in traces:
            session_id = await self.insert_session_trace(trace)
            inserted_ids.append(session_id)
        return inserted_ids
