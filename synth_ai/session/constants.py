"""Constants for agent session usage tracking.

Shared constants used by both SDK and backend.
"""

from __future__ import annotations

# Session status values
SESSION_STATUS_ACTIVE = "active"  # Currently active session (only one per org/user)
SESSION_STATUS_ENDED = "ended"
SESSION_STATUS_LIMIT_EXCEEDED = "limit_exceeded"
SESSION_STATUS_EXPIRED = "expired"

# Limit types
LIMIT_TYPE_HARD = "hard"
LIMIT_TYPE_SOFT = "soft"
LIMIT_TYPE_ROLLING_WINDOW = "rolling_window"

# Metric types
METRIC_TYPE_TOKENS = "tokens"
METRIC_TYPE_COST_USD = "cost_usd"
METRIC_TYPE_GPU_HOURS = "gpu_hours"
METRIC_TYPE_API_CALLS = "api_calls"

# Session types
SESSION_TYPE_API_CALL = "api_call"
SESSION_TYPE_JOB = "job"
SESSION_TYPE_TRACING_SESSION = "tracing_session"

# Reference types for usage records
REFERENCE_TYPE_WALL_CLOCK_EVENT = "wall_clock_event"
REFERENCE_TYPE_LLM_CALL = "llm_call"
REFERENCE_TYPE_API_CALL = "api_call"
REFERENCE_TYPE_CUSTOM = "custom"

# Valid values sets (for validation)
VALID_SESSION_STATUSES = {
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_ENDED,
    SESSION_STATUS_LIMIT_EXCEEDED,
    SESSION_STATUS_EXPIRED,
}

VALID_LIMIT_TYPES = {
    LIMIT_TYPE_HARD,
    LIMIT_TYPE_SOFT,
    LIMIT_TYPE_ROLLING_WINDOW,
}

VALID_METRIC_TYPES = {
    METRIC_TYPE_TOKENS,
    METRIC_TYPE_COST_USD,
    METRIC_TYPE_GPU_HOURS,
    METRIC_TYPE_API_CALLS,
}

VALID_REFERENCE_TYPES = {
    REFERENCE_TYPE_WALL_CLOCK_EVENT,
    REFERENCE_TYPE_LLM_CALL,
    REFERENCE_TYPE_API_CALL,
    REFERENCE_TYPE_CUSTOM,
}

