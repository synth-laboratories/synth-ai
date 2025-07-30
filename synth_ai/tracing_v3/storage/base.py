"""Abstract base class for trace storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

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
    async def get_session_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session trace by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session trace data or None if not found
        """
        pass

    @abstractmethod
    async def query_traces(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.

        Args:
            query: The SQL query to execute
            params: Optional query parameters

        Returns:
            Query results as a DataFrame
        """
        pass

    @abstractmethod
    async def get_model_usage(
        self, start_date: datetime = None, end_date: datetime = None, model_name: str = None
    ) -> pd.DataFrame:
        """Get model usage statistics.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            model_name: Optional model name filter

        Returns:
            Model usage statistics as a DataFrame
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

    # Optional experiment management methods
    async def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str = None,
        configuration: Dict[str, Any] = None,
    ) -> str:
        """Create a new experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    async def link_session_to_experiment(self, session_id: str, experiment_id: str):
        """Link a session to an experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    async def get_sessions_by_experiment(
        self, experiment_id: str, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get all sessions for an experiment."""
        raise NotImplementedError("Experiment management not supported by this backend")

    # Batch operations
    async def batch_insert_sessions(
        self, traces: List[SessionTrace], batch_size: int = 1000
    ) -> List[str]:
        """Batch insert multiple session traces.

        Default implementation calls insert_session_trace for each trace.
        Subclasses can override for more efficient batch operations.
        """
        inserted_ids = []
        for trace in traces:
            session_id = await self.insert_session_trace(trace)
            inserted_ids.append(session_id)
        return inserted_ids
