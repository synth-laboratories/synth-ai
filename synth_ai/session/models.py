"""Agent session data models for SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID


@dataclass
class AgentSessionLimit:
    """Limit configuration for an agent session."""

    limit_type: str  # 'hard', 'soft', 'rolling_window'
    metric_type: str  # 'tokens', 'cost_usd', 'gpu_hours', 'api_calls'
    limit_value: Decimal
    current_usage: Decimal = Decimal("0")
    warning_threshold: Optional[Decimal] = None
    window_seconds: Optional[int] = None
    limit_id: Optional[UUID] = None
    exceeded_at: Optional[datetime] = None
    exceeded_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @property
    def remaining(self) -> Decimal:
        """Calculate remaining limit."""
        return self.limit_value - self.current_usage


@dataclass
class AgentSessionUsage:
    """Current usage summary for a session."""

    tokens: Decimal = Decimal("0")
    cost_usd: Decimal = Decimal("0")
    gpu_hours: Decimal = Decimal("0")
    api_calls: int = 0


@dataclass
class LimitCheckResult:
    """Result of a limit check."""

    allowed: bool
    limit_value: Decimal
    current_usage: Decimal
    remaining: Decimal
    reason: Optional[str] = None


@dataclass
class AgentSession:
    """Agent session representation."""

    session_id: str
    org_id: UUID
    created_at: datetime
    user_id: Optional[UUID] = None
    api_key_id: Optional[UUID] = None
    expires_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str = "active"  # 'active', 'ended', 'limit_exceeded', 'expired'
    limit_exceeded_reason: Optional[str] = None
    usage: AgentSessionUsage = field(default_factory=AgentSessionUsage)
    limits: list[AgentSessionLimit] = field(default_factory=list)
    tracing_session_id: Optional[str] = None
    session_type: str = "api_call"  # 'api_call', 'job', 'tracing_session'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionUsageRecord:
    """Individual usage event within a session."""

    id: UUID
    session_id: str
    metric_type: str
    metric_value: Decimal
    org_id: UUID
    created_at: datetime
    reference_type: Optional[str] = None
    reference_id: Optional[str] = None
    user_id: Optional[UUID] = None
    metadata: dict[str, Any] = field(default_factory=dict)

