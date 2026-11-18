"""Opt-in operation diagnostics containing no request or response payloads."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class DiagnosticPhase(StrEnum):
    RETRY = "retry"
    RETRY_SUPPRESSED = "retry_suppressed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RetrySuppressionReason(StrEnum):
    DEADLINE = "deadline"
    INVALID_RETRY_AFTER = "invalid_retry_after"


@dataclass(frozen=True)
class RetrySuppression:
    reason: RetrySuppressionReason
    proposed_delay_seconds: float | None
    remaining_deadline_seconds: float


@dataclass(frozen=True)
class OperationDiagnostic:
    operation: str
    phase: DiagnosticPhase
    attempts: int
    elapsed_seconds: float
    request_correlation: str | None
    idempotency_correlation: str | None
    retry_suppression: RetrySuppression | None = None


DiagnosticHook = Callable[[OperationDiagnostic], None]
