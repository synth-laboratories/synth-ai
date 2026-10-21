"""Shared bounded retry policy for synchronous and asynchronous reads."""

from __future__ import annotations

import math
import random
import time
from datetime import UTC
from email.utils import parsedate_to_datetime

from synth_ai.errors import RateLimitError, RequestTimeoutError, SynthError

# A longer server hint is reported to the caller, never shortened into a retry.
MAX_RETRY_AFTER_SECONDS = 86400.0


def retry_after_seconds(value: str | None, *, now: float | None = None) -> float | None:
    """Read bounded numeric seconds or an aware HTTP date; past dates mean zero."""
    if value is None or not value.strip() or len(value) > 128:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            seconds = max(
                0.0, date.astimezone(UTC).timestamp() - (time.time() if now is None else now)
            )
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def delay_seconds(
    attempt: int,
    base: float,
    cap: float,
    retry_after: float | None,
    *,
    rng: random.Random,
) -> float:
    """Full jitter with a server floor, capped at one day for delay arithmetic.

    ``retry_failure`` refuses a hint above that cap, so no request sleeps less
    than the server requested. Attempt numbers are one-based and bounded by the
    client's read_retry_count (at most ten retries).
    """
    ceiling = min(cap, base * 2 ** (attempt - 1))
    jitter = rng.uniform(0.0, ceiling)
    return max(jitter, min(retry_after or 0.0, MAX_RETRY_AFTER_SECONDS))


def retry_failure(
    *, status: int | None, delay: float, remaining: float, retry_after: float | None
) -> SynthError | None:
    """Preserve the server hint when a retry cannot fit the shared deadline."""
    if delay < remaining and (retry_after is None or retry_after <= MAX_RETRY_AFTER_SECONDS):
        return None
    if status == 429 and retry_after is not None:
        return RateLimitError(
            "Server retry delay exceeds the request deadline",
            status=status,
            retry_after_seconds=retry_after,
            retryable=True,
        )
    return RequestTimeoutError(
        "Retry would exceed the request deadline",
        status=status,
        retry_after_seconds=retry_after,
    )
