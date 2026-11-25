"""Opt-in operation diagnostics containing no request or response payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DiagnosticHook",
    "DiagnosticPhase",
    "OperationDiagnostic",
    "RetrySuppressed",
    "RetrySuppressedReason",
]


class DiagnosticPhase(StrEnum):
    RETRY = "retry"
    RETRY_SUPPRESSED = "retry_suppressed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RetrySuppressedReason(StrEnum):
    """Why a retryable 429/503 was not retried; the error that follows is the last attempt's."""

    DEADLINE_EXCEEDED = "deadline_exceeded"
    RETRY_AFTER_INVALID = "retry_after_invalid"


@dataclass(frozen=True)
class RetrySuppressed:
    reason: RetrySuppressedReason
    status: int
    retry_after: str | None
    delay_seconds: float | None
    remaining_seconds: float


@dataclass(frozen=True)
class OperationDiagnostic:
    operation: str
    phase: DiagnosticPhase
    attempts: int
    elapsed_seconds: float
    request_correlation: str | None
    idempotency_correlation: str | None
    retry_suppressed: RetrySuppressed | None = None


DiagnosticHook = Callable[[OperationDiagnostic], None]
