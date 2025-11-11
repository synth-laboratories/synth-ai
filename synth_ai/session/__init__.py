"""Agent session module initialization."""

from __future__ import annotations

from synth_ai.session.client import AgentSessionClient
from synth_ai.session.constants import (
    LIMIT_TYPE_HARD,
    LIMIT_TYPE_ROLLING_WINDOW,
    LIMIT_TYPE_SOFT,
    METRIC_TYPE_API_CALLS,
    METRIC_TYPE_COST_USD,
    METRIC_TYPE_GPU_HOURS,
    METRIC_TYPE_TOKENS,
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_ENDED,
    SESSION_STATUS_EXPIRED,
    SESSION_STATUS_LIMIT_EXCEEDED,
    VALID_LIMIT_TYPES,
    VALID_METRIC_TYPES,
    VALID_SESSION_STATUSES,
)
from synth_ai.session.exceptions import (
    InvalidLimitError,
    LimitExceededError,
    SessionNotActiveError,
    SessionNotFoundError,
    SessionUsageError,
)
from synth_ai.session.manager import AgentSessionManager
from synth_ai.session.models import (
    AgentSession,
    AgentSessionLimit,
    AgentSessionUsage,
    LimitCheckResult,
    SessionUsageRecord,
)
from synth_ai.session.query import AgentSessionQuery

__all__ = [
    # Exceptions
    "SessionUsageError",
    "LimitExceededError",
    "SessionNotFoundError",
    "SessionNotActiveError",
    "InvalidLimitError",
    # Models
    "AgentSession",
    "AgentSessionLimit",
    "AgentSessionUsage",
    "LimitCheckResult",
    "SessionUsageRecord",
    # Client
    "AgentSessionClient",
    # Manager
    "AgentSessionManager",
    # Query
    "AgentSessionQuery",
    # Constants
    "SESSION_STATUS_ACTIVE",
    "SESSION_STATUS_ENDED",
    "SESSION_STATUS_LIMIT_EXCEEDED",
    "SESSION_STATUS_EXPIRED",
    "LIMIT_TYPE_HARD",
    "LIMIT_TYPE_SOFT",
    "LIMIT_TYPE_ROLLING_WINDOW",
    "METRIC_TYPE_TOKENS",
    "METRIC_TYPE_COST_USD",
    "METRIC_TYPE_GPU_HOURS",
    "METRIC_TYPE_API_CALLS",
    "VALID_SESSION_STATUSES",
    "VALID_LIMIT_TYPES",
    "VALID_METRIC_TYPES",
]


