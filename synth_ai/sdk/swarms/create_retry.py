"""Bounded retry of a run create the backend definitively did not admit (SYN-4883).

The backend refuses a launch that lost the per-org admission lock with 503
``launch_admission_busy`` and ``outcome=not_accepted``: nothing was admitted, so
resubmitting is safe, but only under a FRESH idempotency key (the refused key's
intent is recorded as not accepted). The same holds for 503 ``owner_unavailable``
(SYN-5091): one owner readiness probe (agents, orchestra or sublinear) failed
before acceptance, the backend recorded ``not_accepted`` and answered with a
bounded ``Retry-After``. Only a ``RefusedError`` whose code is in
``RETRY_SAFE_REFUSAL_CODES`` is retried; every other refusal and every uncertain
outcome surfaces unchanged.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from synth_ai.errors import RefusedError, SynthError
from synth_ai.options import MutationOptions

CREATE_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_AFTER_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 8.0
RETRY_SAFE_REFUSAL_CODES = frozenset({"launch_admission_busy", "owner_unavailable"})

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LaunchAttempt:
    """One refused create attempt."""

    attempt: int
    request_id: str | None
    code: str | None
    cause: str | None
    retry_after_seconds: float | None


class LaunchRetryExhausted(SynthError):  # noqa: N818 - public typed error named for the outcome
    """Every create attempt was refused as not accepted; nothing was admitted.

    ``attempts`` lists each attempt's request id, code and cause, in order.
    """

    def __init__(self, attempts: tuple[LaunchAttempt, ...]) -> None:
        last = attempts[-1]
        ids = ",".join(item.request_id or "-" for item in attempts)
        super().__init__(
            f"Run create refused {len(attempts)} times as not accepted [request_ids={ids}]",
            code=last.code,
            request_id=last.request_id,
            retry_after_seconds=last.retry_after_seconds,
            retryable=True,
            cause=last.cause,
            stage="create_retry",
        )
        self.attempts = attempts

    @property
    def request_ids(self) -> tuple[str | None, ...]:
        return tuple(item.request_id for item in self.attempts)


def is_retry_safe(error: BaseException) -> bool:
    return isinstance(error, RefusedError) and error.code in RETRY_SAFE_REFUSAL_CODES


def retry_delay(error: SynthError) -> float:
    advised = error.retry_after_seconds
    if advised is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    return min(max(advised, 0.0), MAX_RETRY_AFTER_SECONDS)


def fresh_options(mutation: MutationOptions | None) -> MutationOptions:
    """Same options with a new request id and a new (generated) idempotency identity."""
    base = mutation or MutationOptions()
    return base.model_copy(update={"identity": None, "idempotency_key": None, "request_id": None})


def _record(attempts: list[LaunchAttempt], error: SynthError) -> None:
    attempts.append(
        LaunchAttempt(
            attempt=len(attempts) + 1,
            request_id=error.client_request_id or error.request_id,
            code=error.code,
            cause=error.cause,
            retry_after_seconds=error.retry_after_seconds,
        )
    )


def create_with_retry(
    send: Callable[[MutationOptions | None], T],
    mutation: MutationOptions | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempts: list[LaunchAttempt] = []
    options = mutation
    while True:
        try:
            return send(options)
        except RefusedError as error:
            if not is_retry_safe(error):
                raise
            _record(attempts, error)
            if len(attempts) >= CREATE_RETRY_ATTEMPTS:
                raise LaunchRetryExhausted(tuple(attempts)) from error
            sleep(retry_delay(error))
            options = fresh_options(mutation)


async def async_create_with_retry(
    send: Callable[[MutationOptions | None], Awaitable[T]],
    mutation: MutationOptions | None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    attempts: list[LaunchAttempt] = []
    options = mutation
    while True:
        try:
            return await send(options)
        except RefusedError as error:
            if not is_retry_safe(error):
                raise
            _record(attempts, error)
            if len(attempts) >= CREATE_RETRY_ATTEMPTS:
                raise LaunchRetryExhausted(tuple(attempts)) from error
            await sleep(retry_delay(error))
            options = fresh_options(mutation)
