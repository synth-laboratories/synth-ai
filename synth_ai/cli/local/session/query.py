"""Query builder for agent sessions (SDK)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from .client import AgentSessionClient
from .models import AgentSession


class AgentSessionQuery:
    """Fluent query builder for agent sessions."""

    def __init__(self, client: AgentSessionClient):
        self._client = client
        self._filters: dict[str, Any] = {}
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
        self._offset: int = 0

    def org_id(self, org_id: UUID) -> AgentSessionQuery:
        """Filter by organization ID."""
        self._filters["org_id"] = org_id
        return self

    def user_id(self, user_id: UUID) -> AgentSessionQuery:
        """Filter by user ID."""
        self._filters["user_id"] = user_id
        return self

    def status(self, status: str) -> AgentSessionQuery:
        """Filter by status."""
        self._filters["status"] = status
        return self

    def created_after(self, after: datetime) -> AgentSessionQuery:
        """Filter sessions created after date."""
        self._filters["created_after"] = after
        return self

    def created_before(self, before: datetime) -> AgentSessionQuery:
        """Filter sessions created before date."""
        self._filters["created_before"] = before
        return self

    def has_limit_exceeded(self) -> AgentSessionQuery:
        """Filter sessions that exceeded limits."""
        self._filters["limit_exceeded"] = True
        return self

    def tracing_session_id(self, tracing_session_id: str) -> AgentSessionQuery:
        """Filter by tracing session ID."""
        self._filters["tracing_session_id"] = tracing_session_id
        return self

    def order_by(self, field: str, desc: bool = True) -> AgentSessionQuery:
        """Order results."""
        self._order_by = f"{field} {'DESC' if desc else 'ASC'}"
        return self

    def limit(self, limit: int) -> AgentSessionQuery:
        """Limit number of results."""
        self._limit = limit
        return self

    def offset(self, offset: int) -> AgentSessionQuery:
        """Offset for pagination."""
        self._offset = offset
        return self

    async def execute(self) -> list[AgentSession]:
        """Execute query and return results."""
        params: dict[str, Any] = {}
        
        if "org_id" in self._filters:
            # org_id is implicit from API key, so we don't need to pass it
            pass
        if "user_id" in self._filters:
            params["user_id"] = str(self._filters["user_id"])
        if "status" in self._filters:
            params["status"] = self._filters["status"]
        if "created_after" in self._filters:
            params["created_after"] = self._filters["created_after"].isoformat()
        if "created_before" in self._filters:
            params["created_before"] = self._filters["created_before"].isoformat()
        if "limit_exceeded" in self._filters and self._filters["limit_exceeded"]:
            params["status"] = "limit_exceeded"
        if "tracing_session_id" in self._filters:
            params["tracing_session_id"] = self._filters["tracing_session_id"]
        
        if self._limit:
            params["limit"] = self._limit
        if self._offset:
            params["offset"] = self._offset
        
        # Use client's list method
        return await self._client.list(**params)

    async def count(self) -> int:
        """Get count of matching sessions."""
        results = await self.execute()
        return len(results)

    async def first(self) -> Optional[AgentSession]:
        """Get first matching session."""
        results = await self.limit(1).execute()
        return results[0] if results else None

