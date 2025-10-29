from __future__ import annotations

"""
Custom error hierarchy for status CLI commands.
"""



class StatusAPIError(RuntimeError):
    """Raised when the backend returns a non-success response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class StatusCLIError(RuntimeError):
    """Raised for client-side validation errors."""

    pass
