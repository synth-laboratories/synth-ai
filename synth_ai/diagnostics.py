"""Opt-in operation diagnostics containing no request or response payloads."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class DiagnosticPhase(StrEnum):
    RETRY = "retry"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class OperationDiagnostic:
    operation: str
    phase: DiagnosticPhase
    attempts: int
    elapsed_seconds: float
    request_correlation: str | None
    idempotency_correlation: str | None


DiagnosticHook = Callable[[OperationDiagnostic], None]
