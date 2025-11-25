"""Backward compatibility layer for synth_ai.session.

This module has been moved to synth_ai.cli.local.session/.
Imports from this location are deprecated and will be removed in v0.4.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "synth_ai.session is deprecated. Use synth_ai.cli.local.session instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export submodules for backwards compatibility
from synth_ai.cli.local.session import (
    client,
    constants,
    exceptions,
    manager,
    models,
    query,
)

# Re-export everything from the new location
from synth_ai.cli.local.session import (
    AgentSession,
    AgentSessionClient,
    AgentSessionLimit,
    AgentSessionManager,
    AgentSessionQuery,
    AgentSessionUsage,
    InvalidLimitError,
    LIMIT_TYPE_HARD,
    LIMIT_TYPE_ROLLING_WINDOW,
    LIMIT_TYPE_SOFT,
    LimitCheckResult,
    LimitExceededError,
    METRIC_TYPE_API_CALLS,
    METRIC_TYPE_COST_USD,
    METRIC_TYPE_GPU_HOURS,
    METRIC_TYPE_TOKENS,
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_ENDED,
    SESSION_STATUS_EXPIRED,
    SESSION_STATUS_LIMIT_EXCEEDED,
    SessionNotActiveError,
    SessionNotFoundError,
    SessionUsageError,
    SessionUsageRecord,
    VALID_LIMIT_TYPES,
    VALID_METRIC_TYPES,
    VALID_SESSION_STATUSES,
)

__all__ = [
    # Submodules
    "client",
    "constants",
    "exceptions",
    "manager",
    "models",
    "query",
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
