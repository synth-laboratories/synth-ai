"""Errors for status CLI helpers."""

from __future__ import annotations


class StatusAPIError(RuntimeError):
    """Raised when status API calls fail."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
