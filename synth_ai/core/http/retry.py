"""Classified retry policy shared by sync and async transports.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.errors import ContractMismatchError, SynthError
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationMetadata


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts_max: int = 3
    delay_seconds_initial: float = 0.5
    delay_seconds_max: float = 5.0
    retry_statuses: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

    def __post_init__(self) -> None:
        if self.attempts_max < 1:
            raise ValueError("attempts_max must be at least 1")
        if self.delay_seconds_initial < 0 or self.delay_seconds_max < 0:
            raise ValueError("retry delays must be non-negative")
        if self.delay_seconds_initial > self.delay_seconds_max:
            raise ValueError("delay_seconds_initial cannot exceed delay_seconds_max")

    def permits(
        self,
        operation: OperationMetadata,
        *,
        idempotency_key: str | None,
    ) -> bool:
        """Retry only metadata-declared idempotent operations.

        Unsafe methods additionally require an idempotency key.
        """
        if not operation.idempotent:
            return False
        if operation.method is HttpMethod.GET:
            return True
        return bool(idempotency_key)

    def delay_seconds(
        self,
        attempt_index: int,
        *,
        retry_after_seconds: float | None,
    ) -> float:
        """Deterministic exponential backoff capped by policy and Retry-After."""
        if attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        exponential = min(
            self.delay_seconds_max,
            self.delay_seconds_initial * (2**attempt_index),
        )
        if retry_after_seconds is None:
            return exponential
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        return min(self.delay_seconds_max, max(exponential, retry_after_seconds))


def idempotency_key_from_request(request: HttpRequest) -> str | None:
    """Extract an idempotency key from headers or JSON body fields."""
    for header_name in ("Idempotency-Key", "idempotency-key"):
        header_value = request.headers.get(header_name)
        if isinstance(header_value, str) and header_value.strip():
            return header_value.strip()
    body = request.body
    if not isinstance(body, dict):
        return None
    for field_name in ("idempotency_key", "idempotency_key_run_create"):
        value = body.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def should_retry_failure(
    policy: RetryPolicy,
    request: HttpRequest,
    error: BaseException,
) -> bool:
    """Decide whether one failed execute attempt may be retried."""
    if isinstance(error, ContractMismatchError):
        return False
    if not policy.permits(
        request.operation,
        idempotency_key=idempotency_key_from_request(request),
    ):
        return False
    if isinstance(error, SynthError):
        if error.retryable:
            return True
        status = None
        if error.failure is not None:
            status = error.failure.status
        if status is None and hasattr(error, "status"):
            status = getattr(error, "status")
        return isinstance(status, int) and status in policy.retry_statuses
    return False


def retry_after_from_error(error: BaseException) -> float | None:
    if isinstance(error, SynthError):
        return error.retry_after_seconds
    return None


__all__ = [
    "RetryPolicy",
    "idempotency_key_from_request",
    "retry_after_from_error",
    "should_retry_failure",
]
