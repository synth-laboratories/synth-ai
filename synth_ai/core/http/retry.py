"""Classified retry policy shared by sync and async transports.

# See: testing/specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.core.errors import (
    ContractMismatchError,
    RetryDirective,
    RuntimeUnavailableError,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
)
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationMetadata


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts_max: int = 3
    delay_seconds_initial: float = 0.5
    delay_seconds_max: float = 5.0
    retry_statuses: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempts_max, bool)
            or not isinstance(self.attempts_max, int)
            or not 1 <= self.attempts_max <= 100
        ):
            raise ValueError("attempts_max must be an integer from 1 through 100")
        for delay in (self.delay_seconds_initial, self.delay_seconds_max):
            if (
                isinstance(delay, bool)
                or not isinstance(delay, (int, float))
                or not math.isfinite(delay)
            ):
                raise ValueError("retry delays must be finite numbers")
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
        if not math.isfinite(retry_after_seconds) or retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        return min(self.delay_seconds_max, max(exponential, retry_after_seconds))


def idempotency_key_from_request(request: HttpRequest) -> str | None:
    """Extract an idempotency key from headers or JSON body fields."""
    values = [value for name, value in request.headers.items() if name.lower() == "idempotency-key"]
    if isinstance(request.body, dict):
        values.extend(
            request.body[name]
            for name in ("idempotency_key", "idempotency_key_run_create")
            if name in request.body
        )
    identities = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("idempotency identity must be a non-empty string")
        identities.add(value)
    if len(identities) > 1:
        raise ValueError("request contains conflicting idempotency identities")
    return next(iter(identities), None)


def should_retry_failure(
    policy: RetryPolicy,
    request: HttpRequest,
    error: BaseException,
) -> bool:
    """Decide whether one failed execute attempt may be retried."""
    if isinstance(error, (ContractMismatchError, RuntimeUnavailableError)):
        return False
    if not policy.permits(
        request.operation,
        idempotency_key=idempotency_key_from_request(request),
    ):
        return False
    if isinstance(error, SynthError):
        if error.failure is not None:
            # The first-class directive is authoritative, including an explicit false.
            return error.failure.retry.retryable
        if error.retryable:
            return True
        status = None
        if error.failure is not None:
            status = error.failure.status
        if status is None and hasattr(error, "status"):
            status = error.status
        return isinstance(status, int) and status in policy.retry_statuses
    return False


def retry_after_from_error(error: BaseException) -> float | None:
    if isinstance(error, SynthError):
        return error.retry_after_seconds
    return None


# Backend ``core/data/db/backpressure_classification.py::DB_BACKPRESSURE_REASON_CODE``,
# emitted as ``error_code`` by ``core/data/db/admission.py::db_backpressure_payload``.
DB_BACKPRESSURE_ERROR_CODE = "db_backpressure"
DB_BACKPRESSURE_RETRY_EXHAUSTED_CODE = "db_backpressure_retry_exhausted"
_REPLAY_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True, slots=True)
class DbBackpressureRefusal:
    """One typed DB backpressure refusal decoded from a backend error body."""

    status: int
    error_code: str
    component: str | None
    work_class: str | None
    retry_after_seconds: float | None
    operator_action: str | None
    message: str


@dataclass(frozen=True, slots=True)
class DbBackpressureRetryPolicy:
    """Bounds for replaying a request the backend shed before starting DB work.

    Attempts and total sleep are both bounded; the server's ``retry_after`` is
    honored up to ``retry_after_cap_seconds`` and only ever lengthened by jitter.
    """

    attempts_max: int = 5
    total_wait_seconds_max: float = 120.0
    retry_after_cap_seconds: float = 30.0
    delay_seconds_initial: float = 1.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempts_max, bool)
            or not isinstance(self.attempts_max, int)
            or not 1 <= self.attempts_max <= 20
        ):
            raise ValueError("attempts_max must be an integer from 1 through 20")
        for name in (
            "total_wait_seconds_max",
            "retry_after_cap_seconds",
            "delay_seconds_initial",
            "jitter_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.jitter_ratio > 1:
            raise ValueError("jitter_ratio cannot exceed 1")

    def next_delay_seconds(
        self,
        *,
        attempts_made: int,
        waited_seconds: float,
        refusal: DbBackpressureRefusal,
        rng: random.Random,
    ) -> float | None:
        """Return the sleep before the next attempt, or None when bounds are spent."""
        if attempts_made >= self.attempts_max:
            return None
        if refusal.retry_after_seconds is not None:
            base = min(self.retry_after_cap_seconds, refusal.retry_after_seconds)
        else:
            base = min(
                self.retry_after_cap_seconds,
                self.delay_seconds_initial * (2 ** (attempts_made - 1)),
            )
        delay = base * (1.0 + rng.uniform(0.0, self.jitter_ratio))
        if waited_seconds + delay > self.total_wait_seconds_max:
            return None
        return delay


def request_permits_replay(method: str, headers: Mapping[str, str]) -> bool:
    """A request may be replayed only when safe by method or carrying an idempotency key."""
    if method.upper() in _REPLAY_SAFE_METHODS:
        return True
    return any(
        name.lower() == "idempotency-key" and bool(str(value).strip())
        for name, value in headers.items()
    )


class DbBackpressureRetryExhaustedError(SynthError):
    """The backend kept shedding DB work past the retry bounds.

    Carries the last refusal's ``component``, ``work_class`` and ``retry_after``.
    Not retryable again: callers already spent the bounded budget.
    """

    def __init__(
        self,
        *,
        method: str,
        path: str,
        operation_id: str | None,
        refusal: DbBackpressureRefusal,
        attempts: int,
        waited_seconds: float,
    ) -> None:
        self.method = method
        self.path = path
        self.refusal = refusal
        self.attempts = attempts
        self.waited_seconds = waited_seconds
        detail = (
            f"{method} {path}: backend refused DB work ({refusal.error_code}) on "
            f"{attempts} attempts over {waited_seconds:.1f}s; last component="
            f"{refusal.component} work_class={refusal.work_class} "
            f"retry_after={refusal.retry_after_seconds} status={refusal.status} "
            f"operator_action={refusal.operator_action}"
        )
        super().__init__(
            detail,
            failure=SynthFailure(
                code=SynthErrorCode(DB_BACKPRESSURE_RETRY_EXHAUSTED_CODE),
                category=SynthErrorCategory.TRANSIENT_SERVICE,
                operation=operation_id,
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(
                    retryable=False,
                    retry_after_seconds=refusal.retry_after_seconds,
                ),
                status=refusal.status,
                detail=detail,
                reason=refusal.component,
            ),
        )

    @property
    def component(self) -> str | None:
        return self.refusal.component

    @property
    def work_class(self) -> str | None:
        return self.refusal.work_class

    @property
    def retry_after_seconds(self) -> float | None:
        return self.refusal.retry_after_seconds


__all__ = [
    "DB_BACKPRESSURE_ERROR_CODE",
    "DbBackpressureRefusal",
    "DbBackpressureRetryExhaustedError",
    "DbBackpressureRetryPolicy",
    "request_permits_replay",
    "RetryPolicy",
    "idempotency_key_from_request",
    "retry_after_from_error",
    "should_retry_failure",
]
