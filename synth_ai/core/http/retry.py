"""Classified retry policy shared by sync and async transports."""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.http.request import HttpMethod


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

    def permits(self, method: HttpMethod, *, idempotency_key: str | None) -> bool:
        return method in {HttpMethod.GET, HttpMethod.PUT, HttpMethod.DELETE} or bool(
            idempotency_key
        )


__all__ = ["RetryPolicy"]
