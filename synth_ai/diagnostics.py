"""Opt-in operation diagnostics containing no request or response payloads."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class DiagnosticPhase(StrEnum):
    RETRY_SUPPRESSED = "retry_suppressed"
    RETRY = "retry"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RetrySuppressionReason(StrEnum):
    DEADLINE = "deadline"


@dataclass(frozen=True)
class SuppressedRetry:
    reason: RetrySuppressionReason
    delay_seconds: float
    remaining_seconds: float


@dataclass(frozen=True)
class OperationDiagnostic:
    operation: str
    phase: DiagnosticPhase
    attempts: int
    elapsed_seconds: float
    request_correlation: str | None
    idempotency_correlation: str | None
    suppressed_retry: SuppressedRetry | None = None


DiagnosticHook = Callable[[OperationDiagnostic], None]
