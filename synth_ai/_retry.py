"""Bounded retry policy shared by operations and run observation."""

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


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
        return max(server_delay if value is not None else 0, backoff) + random.uniform(
            0, self.jitter_seconds
        )


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
    return max(0, seconds) if math.isfinite(seconds) and seconds <= 60 else None
