"""Storage interfaces for Synth AI SDK.

This module provides abstract interfaces for storage backends,
allowing different storage implementations (SQLite, Postgres, etc.)
to be used interchangeably.

Note: The actual storage implementations remain in tracing_v3/storage/
to avoid breaking changes. This module provides type hints and interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_ai.data.traces import SessionTrace


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def store_trace(self, trace: SessionTrace) -> str:
        """Store a session trace.

        Args:
            trace: The session trace to store

        Returns:
            The session ID of the stored trace
        """
        pass

    @abstractmethod
    async def get_trace(self, session_id: str) -> SessionTrace | None:
        """Retrieve a session trace by ID.

        Args:
            session_id: The session ID to look up

        Returns:
            The session trace, or None if not found
        """
        pass

    @abstractmethod
    async def list_sessions(
        self,
        experiment_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List session summaries.

        Args:
            experiment_id: Optional filter by experiment
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of session summary dicts
        """
        pass


__all__ = [
    "StorageBackend",
]


