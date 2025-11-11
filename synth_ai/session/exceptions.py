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
        super().__init__(
            f"Session {session_id} {metric_type} limit exceeded: "
            f"limit={limit_value}, current={current_usage}, attempted={attempted_usage}"
        )


class SessionNotFoundError(SessionUsageError):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session {session_id} not found")


class SessionNotActiveError(SessionUsageError):
    """Raised when attempting to use an inactive session."""

    def __init__(self, session_id: str, status: str):
        self.session_id = session_id
        self.status = status
        super().__init__(f"Session {session_id} is not active (status: {status})")


class InvalidLimitError(SessionUsageError):
    """Raised when limit configuration is invalid."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"Invalid limit configuration: {message}")

