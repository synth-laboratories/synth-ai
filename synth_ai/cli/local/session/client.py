"""AgentSession API client for SDK."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List, Optional
from uuid import UUID

from synth_ai.core.http import AsyncHttpClient, HTTPError

from .exceptions import (
    InvalidLimitError,
    SessionNotActiveError,
    SessionNotFoundError,
)
from .models import (
    AgentSession,
    AgentSessionLimit,
    AgentSessionUsage,
    LimitCheckResult,
    SessionUsageRecord,
)


class AgentSessionClient:
    """Client for managing agent sessions via API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        http: Optional[AsyncHttpClient] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._http = http or AsyncHttpClient(base_url, api_key, timeout)

    def _parse_session(self, data: dict[str, Any]) -> AgentSession:
        """Parse session from API response."""
        usage_data = data.get("current_usage", {})
        usage = AgentSessionUsage(
            tokens=Decimal(str(usage_data.get("tokens", 0))),
            cost_usd=Decimal(str(usage_data.get("cost_usd", 0))),
            gpu_hours=Decimal(str(usage_data.get("gpu_hours", 0))),
            api_calls=int(usage_data.get("api_calls", 0)),
        )

        limits = []
        for limit_data in data.get("limits", []):
            limits.append(
                AgentSessionLimit(
                    limit_id=UUID(limit_data["limit_id"]) if limit_data.get("limit_id") else None,
                    limit_type=limit_data["limit_type"],
                    metric_type=limit_data["metric_type"],
                    limit_value=Decimal(str(limit_data["limit_value"])),
                    current_usage=Decimal(str(limit_data.get("current_usage", 0))),
                    warning_threshold=Decimal(str(limit_data["warning_threshold"]))
                    if limit_data.get("warning_threshold")
                    else None,
                    window_seconds=limit_data.get("window_seconds"),
                    exceeded_at=datetime.fromisoformat(limit_data["exceeded_at"].replace("Z", "+00:00"))
                    if limit_data.get("exceeded_at")
                    else None,
                    exceeded_reason=limit_data.get("exceeded_reason"),
                    created_at=datetime.fromisoformat(limit_data["created_at"].replace("Z", "+00:00"))
                    if limit_data.get("created_at")
                    else None,
                    expires_at=datetime.fromisoformat(limit_data["expires_at"].replace("Z", "+00:00"))
                    if limit_data.get("expires_at")
                    else None,
                )
            )

        return AgentSession(
            session_id=data["session_id"],
            org_id=UUID(data["org_id"]),
            user_id=UUID(data["user_id"]) if data.get("user_id") else None,
            api_key_id=UUID(data["api_key_id"]) if data.get("api_key_id") else None,
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
            if data.get("expires_at")
            else None,
            ended_at=datetime.fromisoformat(data["ended_at"].replace("Z", "+00:00"))
            if data.get("ended_at")
            else None,
            status=data["status"],
            limit_exceeded_reason=data.get("limit_exceeded_reason"),
            usage=usage,
            limits=limits,
            tracing_session_id=data.get("tracing_session_id"),
            session_type=data.get("session_type", "api_call"),
            metadata=data.get("metadata", {}),
        )

    async def create(
        self,
        org_id: Optional[UUID] = None,
        limits: Optional[List[dict[str, Any]]] = None,
        tracing_session_id: Optional[str] = None,
        session_type: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        make_active: bool = True,  # If True, ends existing active session and makes this one active
    ) -> AgentSession:
        """Create a new agent session.
        
        If org_id is not provided, it will be fetched from /api/v1/me endpoint.
        """
        # Get org_id from /me endpoint if not provided
        if org_id is None:
            try:
                async with self._http:
                    me_data = await self._http.get("/api/v1/me")
                    org_id = UUID(me_data["org_id"])
            except Exception as e:
                raise ValueError(
                    f"Failed to get org_id from API endpoint '/api/v1/me': {e}. "
                    f"Please provide org_id explicitly when creating a session, "
                    f"or ensure your API key is valid and has access to the /api/v1/me endpoint."
                ) from e
        
        payload: dict[str, Any] = {}
        if org_id:
            payload["org_id"] = str(org_id)
        if limits:
            payload["limits"] = limits
        if tracing_session_id:
            payload["tracing_session_id"] = tracing_session_id
        if session_type:
            payload["session_type"] = session_type
        if expires_at:
            payload["expires_at"] = expires_at.isoformat()
        if metadata:
            payload["metadata"] = metadata
        if session_id:
            payload["session_id"] = session_id
        if make_active is not None:
            payload["make_active"] = make_active

        try:
            async with self._http:
                data = await self._http.post_json("/api/v1/sessions", json=payload)
                return self._parse_session(data)
        except HTTPError as e:
            if e.status == 400:
                raise InvalidLimitError(str(e)) from e
            raise

    async def get(self, session_id: str) -> AgentSession:
        """Get session by ID."""
        try:
            async with self._http:
                data = await self._http.get(f"/api/v1/sessions/{session_id}")
                return self._parse_session(data)
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            raise

    async def end(self, session_id: str) -> AgentSession:
        """End a session."""
        try:
            async with self._http:
                data = await self._http.post_json(f"/api/v1/sessions/{session_id}/end", json={})
                return self._parse_session(data)
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            if e.status == 400:
                raise SessionNotActiveError(session_id, "unknown") from e
            raise

    async def check_limit(
        self,
        session_id: str,
        metric_type: str,
        requested_value: Decimal,
    ) -> LimitCheckResult:
        """Check if usage would exceed limits."""
        # Note: This endpoint doesn't exist yet, but could be added
        # For now, we'll get the session and check limits client-side
        session = await self.get(session_id)
        if session.status != "active":
            return LimitCheckResult(
                allowed=False,
                limit_value=Decimal("0"),
                current_usage=Decimal("0"),
                remaining=Decimal("0"),
                reason=f"Session not active (status: {session.status})",
            )

        # Find relevant limits
        for limit in session.limits:
            if limit.metric_type == metric_type:
                projected = limit.current_usage + requested_value
                if projected > limit.limit_value:
                    return LimitCheckResult(
                        allowed=False,
                        limit_value=limit.limit_value,
                        current_usage=limit.current_usage,
                        remaining=limit.remaining,
                        reason=f"Limit exceeded: {projected} > {limit.limit_value}",
                    )

        return LimitCheckResult(
            allowed=True,
            limit_value=Decimal("0"),
            current_usage=Decimal("0"),
            remaining=Decimal("999999999"),
            reason=None,
        )

    async def increase_limit(
        self,
        session_id: str,
        metric_type: str,
        increase_by: Decimal,
    ) -> AgentSessionLimit:
        """Increase a limit by amount."""
        try:
            async with self._http:
                data = await self._http.post_json(
                    f"/api/v1/sessions/{session_id}/limits/increase",
                    json={"metric_type": metric_type, "increase_by": float(increase_by)},
                )
                return AgentSessionLimit(
                    limit_id=UUID(data["limit_id"]) if data.get("limit_id") else None,
                    limit_type=data["limit_type"],
                    metric_type=data["metric_type"],
                    limit_value=Decimal(str(data["limit_value"])),
                    current_usage=Decimal(str(data.get("current_usage", 0))),
                    warning_threshold=Decimal(str(data["warning_threshold"]))
                    if data.get("warning_threshold")
                    else None,
                    window_seconds=data.get("window_seconds"),
                    exceeded_at=datetime.fromisoformat(data["exceeded_at"].replace("Z", "+00:00"))
                    if data.get("exceeded_at")
                    else None,
                    exceeded_reason=data.get("exceeded_reason"),
                    created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                    if data.get("created_at")
                    else None,
                    expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                    if data.get("expires_at")
                    else None,
                )
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            if e.status == 400:
                raise SessionNotActiveError(session_id, "unknown") from e
            raise

    async def add_limit(
        self,
        session_id: str,
        limit: dict[str, Any],
    ) -> AgentSessionLimit:
        """Add a new limit to session."""
        try:
            async with self._http:
                data = await self._http.post_json(
                    f"/api/v1/sessions/{session_id}/limits",
                    json=limit,
                )
                return AgentSessionLimit(
                    limit_id=UUID(data["limit_id"]) if data.get("limit_id") else None,
                    limit_type=data["limit_type"],
                    metric_type=data["metric_type"],
                    limit_value=Decimal(str(data["limit_value"])),
                    current_usage=Decimal(str(data.get("current_usage", 0))),
                    warning_threshold=Decimal(str(data["warning_threshold"]))
                    if data.get("warning_threshold")
                    else None,
                    window_seconds=data.get("window_seconds"),
                    exceeded_at=datetime.fromisoformat(data["exceeded_at"].replace("Z", "+00:00"))
                    if data.get("exceeded_at")
                    else None,
                    exceeded_reason=data.get("exceeded_reason"),
                    created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                    if data.get("created_at")
                    else None,
                    expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                    if data.get("expires_at")
                    else None,
                )
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            if e.status == 400:
                raise InvalidLimitError(str(e)) from e
            raise

    async def list_limits(self, session_id: str) -> List[AgentSessionLimit]:
        """Get all limits for a session."""
        try:
            async with self._http:
                data = await self._http.get(f"/api/v1/sessions/{session_id}/limits")
                return [
                    AgentSessionLimit(
                        limit_id=UUID(limit_data["limit_id"]) if limit_data.get("limit_id") else None,
                        limit_type=limit_data["limit_type"],
                        metric_type=limit_data["metric_type"],
                        limit_value=Decimal(str(limit_data["limit_value"])),
                        current_usage=Decimal(str(limit_data.get("current_usage", 0))),
                        warning_threshold=Decimal(str(limit_data["warning_threshold"]))
                        if limit_data.get("warning_threshold")
                        else None,
                        window_seconds=limit_data.get("window_seconds"),
                        exceeded_at=datetime.fromisoformat(limit_data["exceeded_at"].replace("Z", "+00:00"))
                        if limit_data.get("exceeded_at")
                        else None,
                        exceeded_reason=limit_data.get("exceeded_reason"),
                        created_at=datetime.fromisoformat(limit_data["created_at"].replace("Z", "+00:00"))
                        if limit_data.get("created_at")
                        else None,
                        expires_at=datetime.fromisoformat(limit_data["expires_at"].replace("Z", "+00:00"))
                        if limit_data.get("expires_at")
                        else None,
                    )
                    for limit_data in data
                ]
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            raise

    async def list(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AgentSession]:
        """List sessions for the authenticated organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        try:
            async with self._http:
                data = await self._http.get("/api/v1/sessions", params=params)
                return [self._parse_session(session_data) for session_data in data]
        except HTTPError:
            raise

    async def get_usage(
        self,
        session_id: str,
        metric_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionUsageRecord]:
        """Get usage records for a session."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if metric_type:
            params["metric_type"] = metric_type

        try:
            async with self._http:
                data = await self._http.get(f"/api/v1/sessions/{session_id}/usage", params=params)
                return [
                    SessionUsageRecord(
                        id=UUID(record_data["id"]),
                        session_id=record_data["session_id"],
                        metric_type=record_data["metric_type"],
                        metric_value=Decimal(str(record_data["metric_value"])),
                        reference_type=record_data.get("reference_type"),
                        reference_id=record_data.get("reference_id"),
                        org_id=UUID(record_data["org_id"]),
                        user_id=UUID(record_data["user_id"]) if record_data.get("user_id") else None,
                        metadata=record_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(record_data["created_at"].replace("Z", "+00:00")),
                    )
                    for record_data in data
                ]
        except HTTPError as e:
            if e.status == 404:
                raise SessionNotFoundError(session_id) from e
            raise
