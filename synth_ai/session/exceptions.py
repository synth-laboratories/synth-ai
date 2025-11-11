"""Shared exceptions for agent session usage tracking.

These exceptions are used by both the SDK (synth-ai) and backend (monorepo).
"""

from __future__ import annotations

from decimal import Decimal


class SessionUsageError(Exception):
    """Base exception for session usage errors."""

    pass


class LimitExceededError(SessionUsageError):
    """Raised when a session limit is exceeded."""

    def __init__(
        self,
        session_id: str,
        metric_type: str,
        limit_value: Decimal,
        current_usage: Decimal,
        attempted_usage: Decimal,
        reason: str | None = None,
    ):
        self.session_id = session_id
        self.metric_type = metric_type
        self.limit_value = limit_value
        self.current_usage = current_usage
        self.attempted_usage = attempted_usage
        self.reason = reason or "Hard limit exceeded"
        
        # Format metric type for display
        metric_display = {
            "cost_usd": "cost",
            "gpu_hours": "GPU hours",
            "api_calls": "API calls",
        }.get(metric_type, metric_type)
        
        # Calculate what would have been the new total
        projected_total = current_usage + attempted_usage
        remaining = limit_value - current_usage
        
        message = (
            f"Session '{session_id}' {metric_display} limit exceeded. "
            f"Limit: {limit_value}, Current usage: {current_usage}, "
            f"Attempted: {attempted_usage} (would be {projected_total} total). "
            f"Remaining budget: {remaining}. "
            f"To continue, end this session and create a new one with a higher limit, "
            f"or increase the limit using client.increase_limit()"
        )
        super().__init__(message)


class SessionNotFoundError(SessionUsageError):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        message = (
            f"Session '{session_id}' not found. "
            f"This session may have been deleted, or you may not have access to it. "
            f"Verify the session ID is correct and belongs to your organization. "
            f"List your sessions using client.list()"
        )
        super().__init__(message)


class SessionNotActiveError(SessionUsageError):
    """Raised when attempting to use an inactive session."""

    def __init__(self, session_id: str, status: str):
        self.session_id = session_id
        self.status = status
        
        status_messages = {
            "ended": "has been ended",
            "limit_exceeded": "has exceeded its limit",
            "expired": "has expired",
        }
        status_desc = status_messages.get(status, f"has status '{status}'")
        
        message = (
            f"Session '{session_id}' is not active - it {status_desc}. "
            f"To use this session, you must create a new active session using client.create()"
        )
        super().__init__(message)


class InvalidLimitError(SessionUsageError):
    """Raised when limit configuration is invalid."""

    def __init__(self, message: str):
        self.message = message
        full_message = (
            f"Invalid limit configuration: {message}. "
            f"Valid limit types: 'hard', 'soft', 'rolling_window'. "
            f"Valid metric types: 'tokens', 'cost_usd', 'gpu_hours', 'api_calls'. "
            f"Limit value must be positive (> 0)."
        )
        super().__init__(full_message)

