"""High-level AgentSession manager for SDK."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from .client import AgentSessionClient
from .exceptions import LimitExceededError, SessionNotFoundError
from .models import AgentSession


class AgentSessionManager:
    """High-level manager for agent sessions.

    Provides convenient methods for common operations and context managers
    for automatic session lifecycle management.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        org_id: Optional[UUID] = None,
        default_limits: Optional[list[dict[str, Any]]] = None,
    ):
        self.client = AgentSessionClient(base_url, api_key)
        self.org_id = org_id
        self.default_limits = default_limits or []
        self._current_session: Optional[AgentSession] = None

    @asynccontextmanager
    async def session(
        self,
        session_id: Optional[str] = None,
        limits: Optional[list[dict[str, Any]]] = None,
        tracing_session_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
        auto_end: bool = True,
    ):
        """Context manager for agent session lifecycle.

        Example:
            async with manager.session(limits=[...]) as session:
                # Use session.session_id in API calls
                await manager.record_usage(session.session_id, ...)
        """
        if not self.org_id:
            raise ValueError(
                "org_id is required for session creation. "
                "Either provide org_id explicitly, or ensure the client can fetch it from /api/v1/me endpoint."
            )

        limits = limits or self.default_limits
        session = await self.client.create(
            org_id=self.org_id,
            limits=limits,
            tracing_session_id=tracing_session_id,
            expires_at=expires_at,
            metadata=metadata,
            session_id=session_id,
        )

        self._current_session = session
        try:
            yield session
        finally:
            if auto_end:
                await self.client.end(session.session_id)
            self._current_session = None

    async def ensure_session(
        self,
        session_id: Optional[str] = None,
    ) -> AgentSession:
        """Get existing session or create new one.

        Useful for CLI where session might already exist.
        """
        if session_id:
            try:
                return await self.client.get(session_id)
            except SessionNotFoundError:
                pass

        # Create new session
        if not self.org_id:
            raise ValueError(
                "org_id is required for session creation. "
                "Either provide org_id explicitly, or ensure the client can fetch it from /api/v1/me endpoint."
            )

        return await self.client.create(
            org_id=self.org_id,
            limits=self.default_limits,
        )

    @property
    def current_session(self) -> Optional[AgentSession]:
        """Get current active session (if any)."""
        return self._current_session

    async def check_and_record(
        self,
        session_id: str,
        metric_type: str,
        estimated_value: Decimal,
        actual_value: Decimal,
        reference_type: Optional[str] = None,
        reference_id: Optional[str] = None,
    ) -> None:
        """Check limit, record usage, handle errors.

        Convenience method that:
        1. Checks limit with estimated value
        2. Records actual usage (if limit check endpoint exists)
        3. Raises LimitExceededError if limit hit

        Note: Actual usage recording would need to be implemented via API endpoint.
        """
        # Check limit
        check = await self.client.check_limit(session_id, metric_type, estimated_value)
        if not check.allowed:
            raise LimitExceededError(
                session_id=session_id,
                metric_type=metric_type,
                limit_value=check.limit_value,
                current_usage=check.current_usage,
                attempted_usage=estimated_value,
                reason=check.reason,
            )

        # Note: Usage recording would need an API endpoint
        # For now, this is a placeholder

