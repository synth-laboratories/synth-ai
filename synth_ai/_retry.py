"""Bounded retry policy shared by operations and run observation."""

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from ._generated.models import Problem

# Servers that predate contract 1.4.0's typed `retryable` Problem member still mark an event
# owner handoff (for example during a deploy) only by this code and status.
_LEGACY_RETRYABLE_STREAM_REFUSALS = frozenset({("event_owner_unavailable", 503)})


@dataclass(frozen=True)
class RetryPolicy:
    """Bound one operation; ambiguous mutations are never automatically replayed."""

    attempts: int = 3
    deadline_seconds: float = 90.0
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 8.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or not 1 <= self.attempts <= 6:
            raise ValueError("retry attempts must be between 1 and 6")
        values = (
            self.deadline_seconds,
            self.backoff_seconds,
            self.backoff_max_seconds,
            self.jitter_seconds,
        )
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or self.deadline_seconds == 0
        ):
            raise ValueError(
                "retry durations must be finite and nonnegative; deadline must be positive"
            )

    def delay(self, value: str | None, attempt: int) -> float | None:
        server_delay = retry_delay(value, attempt)
        if server_delay is None:
            return None
        backoff = min(self.backoff_max_seconds, self.backoff_seconds * float(2**attempt))
        jitter = random.uniform(0, self.jitter_seconds)  # noqa: S311 -- non-cryptographic backoff jitter
        return max(server_delay if value is not None else 0, backoff) + jitter


def retry_delay(value: str | None, attempt: int) -> float | None:
    if value is None:
        return float(2**attempt)
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


def stream_refusal_retryable(problem: Problem) -> bool:
    """Whether an event-feed refusal permits a checkpointed reconnect.

    Only the server decides: an explicit `retryable` member is authoritative; without
    one, only the exact legacy owner-handoff refusal qualifies. Everything else (authentication, cursor expiry,
    404, contract errors and local SDK problems) propagates unchanged.
    """
    if problem.service == "synth-ai":
        return False
    if problem.retryable is not None:
        return problem.retryable
    return (problem.code, problem.status) in _LEGACY_RETRYABLE_STREAM_REFUSALS


# A send to a run whose admission is not yet committed is refused with this typed 503
# before Core holds the body (SYN-5744). The refusal is retried under the same
# Idempotency-Key, outside the operation's attempt and deadline bounds, for at most
# this total budget; each wait is the server's Retry-After, capped.
ADMISSION_PENDING_CODE = "run_message_admission_pending"
ADMISSION_RETRY_BUDGET_SECONDS = 120.0
ADMISSION_RETRY_WAIT_DEFAULT_SECONDS = 1.0
ADMISSION_RETRY_WAIT_MAX_SECONDS = 60.0


def admission_pending(problem: Problem | None, status: int) -> bool:
    """Whether a refusal is Core's typed not-yet-admitted send refusal."""
    return (
        problem is not None
        and status == 503
        and problem.status == 503
        and problem.code == ADMISSION_PENDING_CODE
        and problem.retryable is not False
    )


def admission_wait(value: str | None) -> float:
    """Retry-After seconds (or an HTTP date); missing or invalid waits 1 s; capped at 60 s."""
    delay = retry_delay(value, 0) if value is not None else None
    if delay is None:
        delay = ADMISSION_RETRY_WAIT_DEFAULT_SECONDS
    return min(ADMISSION_RETRY_WAIT_MAX_SECONDS, delay)
