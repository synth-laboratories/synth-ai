"""Per-response rate capacity; see testing/the public contract snapshots in contracts/

Only recognized, bounded numeric/ISO fields are retained. Response cookies,
authorization, and unrelated headers never enter logs or exception metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    dimension: str
    limit: int | None
    remaining: int | None
    reset_at: str | None
    burst: int | None = None


def parse_rate_limits(headers: Mapping[str, str]) -> tuple[RateLimitSnapshot, ...]:
    normalized = {name.lower(): value for name, value in headers.items()}
    result = []
    for dimension in ("requests", "concurrency", "tokens", "spend-cents"):

        def count(kind: str, dimension: str = dimension) -> int | None:
            raw = normalized.get(f"x-ratelimit-{kind}-{dimension}", "")
            return (
                int(raw)
                if raw.isascii() and raw.isdigit() and len(raw) <= 16 and int(raw) <= 2**53 - 1
                else None
            )

        limit, remaining, burst = count("limit"), count("remaining"), count("burst")
        reset = normalized.get(f"x-ratelimit-reset-{dimension}")
        if reset is not None:
            try:
                if (
                    len(reset) > 64
                    or datetime.fromisoformat(reset.replace("Z", "+00:00")).tzinfo is None
                ):
                    reset = None
            except ValueError:
                reset = None
        if limit is not None or remaining is not None or reset is not None or burst is not None:
            result.append(RateLimitSnapshot(dimension, limit, remaining, reset, burst))
    return tuple(result)


@dataclass(slots=True)
class ResponseMetadata:
    """Caller-owned metadata for one request; safe for concurrent transports.

    Pass a distinct instance per concurrent request. Metadata is replaced on
    each response, so a later response without rate headers cannot reuse old
    capacity. JSON/result wire contracts remain unchanged.
    """

    rate_limits: tuple[RateLimitSnapshot, ...] = ()

    def capture(self, headers: Mapping[str, str]) -> None:
        self.rate_limits = parse_rate_limits(headers)


def describe_rate_limit(snapshot: RateLimitSnapshot) -> str:
    """One human line per dimension, e.g. ``rate limit concurrency: remaining 3/limit 5, resets …``."""

    def known(value: object) -> str:
        return "unknown" if value is None else str(value)

    line = (
        f"rate limit {snapshot.dimension}: remaining {known(snapshot.remaining)}"
        f"/limit {known(snapshot.limit)}, resets {known(snapshot.reset_at)}"
    )
    return line if snapshot.burst is None else f"{line}, burst {snapshot.burst}"
