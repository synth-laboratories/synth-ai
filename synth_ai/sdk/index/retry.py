"""Bounded retry of uncertain durable Search requests.

A DEEP Search is admitted (funds held, a concurrency slot taken) before the
response that reports it is delivered, so a transient failure on create does
not mean "nothing happened". Two rules keep a retry from creating or paying
for a second Search:

* **Same request, same Idempotency-Key.** Creation always sends a key; a retry
  replays it and the backend returns the Search it already admitted.
* **Reconnect when the failure names the Search.** A failure that carries the
  admitted Search ID (``X-Synth-Search-Id`` header or ``search_id`` in the
  error body) is resolved by reading that Search, not by creating again.

Only transient service failures (HTTP 502/503/504, transport timeouts and
network errors) are retried. A ``Retry-After`` longer than the remaining wait
budget is surfaced to the caller instead of slept through: a platform daily
budget that resets at midnight must not block a program for hours.
"""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.errors import HTTPError, SynthError, SynthErrorCategory
from synth_ai.core.http.transport import SEARCH_RESOURCE_KIND

_TRANSIENT_STATUSES = frozenset({502, 503, 504})


@dataclass(frozen=True, slots=True)
class IndexRetryPolicy:
    """Attempts and waits for one logical Search request.

    ``total_wait_seconds_max`` bounds the sleeping between attempts, not the
    server's work: a DEEP Search itself may run for its full deadline.
    """

    attempts_max: int = 3
    delay_seconds_initial: float = 1.0
    delay_seconds_max: float = 10.0
    total_wait_seconds_max: float = 30.0

    def __post_init__(self) -> None:
        if self.attempts_max < 1:
            raise ValueError("attempts_max must be at least 1")
        if (
            min(
                self.delay_seconds_initial,
                self.delay_seconds_max,
                self.total_wait_seconds_max,
            )
            < 0
        ):
            raise ValueError("retry delays must be non-negative")

    def next_delay(
        self, error: BaseException, *, attempt_index: int, waited_seconds: float
    ) -> float | None:
        """Seconds to wait before retrying, or ``None`` to raise ``error`` now."""
        if not is_transient_search_failure(error):
            return None
        if attempt_index + 1 >= self.attempts_max:
            return None
        remaining = self.total_wait_seconds_max - waited_seconds
        exponential = min(self.delay_seconds_max, self.delay_seconds_initial * (2**attempt_index))
        retry_after = error.retry_after_seconds if isinstance(error, SynthError) else None
        if retry_after is not None and retry_after > remaining:
            return None
        delay = max(exponential, retry_after or 0.0)
        return max(0.0, min(delay, remaining))


DEFAULT_INDEX_RETRY_POLICY = IndexRetryPolicy()


def is_transient_search_failure(error: BaseException) -> bool:
    """Whether the request may be replayed unchanged with the same key."""
    if not isinstance(error, SynthError) or error.failure is None:
        return False
    if error.failure.category is not SynthErrorCategory.TRANSIENT_SERVICE:
        return False
    status = error.failure.status
    return status is None or status == 0 or status in _TRANSIENT_STATUSES


def search_id_from_error(error: BaseException) -> str | None:
    """The admitted Search a failure names, if the server named one."""
    if not isinstance(error, SynthError):
        return None
    resource = error.resource
    if resource is not None and resource.kind == SEARCH_RESOURCE_KIND:
        return resource.resource_id
    if isinstance(error, HTTPError) and isinstance(error.detail, dict):
        for source in (error.detail.get("detail"), error.detail):
            if isinstance(source, dict):
                value = source.get("search_id")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


__all__ = [
    "DEFAULT_INDEX_RETRY_POLICY",
    "IndexRetryPolicy",
    "is_transient_search_failure",
    "search_id_from_error",
]
