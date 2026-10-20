"""Run create retries a definite launch_admission_busy under fresh keys (SYN-4883)."""

from __future__ import annotations

import asyncio

import pytest
from synth_ai.errors import ConflictError, OutcomeUncertainError, RefusedError
from synth_ai.options import MutationOptions
from synth_ai.sdk.swarms.create_retry import (
    CREATE_RETRY_ATTEMPTS,
    DEFAULT_RETRY_AFTER_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    LaunchRetryExhausted,
    async_create_with_retry,
    create_with_retry,
)

pytestmark = pytest.mark.unit


def busy(request_id: str, retry_after: float | None = 1.0) -> RefusedError:
    error = RefusedError(
        "create refused: launch_admission_busy",
        code="launch_admission_busy",
        status=503,
        retry_after_seconds=retry_after,
        retryable=False,
        cause="admission_lock_wait_exhausted:8000",
    )
    error.client_request_id = request_id
    return error


class FakeTransport:
    """Scripted outcomes; records the options each attempt was sent with."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[MutationOptions | None] = []

    def send(self, options: MutationOptions | None) -> str:
        self.sent.append(options)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    async def async_send(self, options: MutationOptions | None) -> str:
        return self.send(options)


def test_busy_then_success_retries_with_a_fresh_key_and_honors_retry_after() -> None:
    transport = FakeTransport(busy("r1", 2.5), "run-1")
    sleeps: list[float] = []
    pinned = MutationOptions(idempotency_key="caller-key", request_id="r1")
    assert create_with_retry(transport.send, pinned, sleep=sleeps.append) == "run-1"
    assert sleeps == [2.5]
    assert transport.sent[0] is pinned
    retry = transport.sent[1]
    assert retry is not None
    assert retry.idempotency_key is None and retry.identity is None
    assert retry.request_id is None


def test_retry_after_defaults_and_caps() -> None:
    transport = FakeTransport(busy("a", None), busy("b", 60.0), "ok")
    sleeps: list[float] = []
    create_with_retry(transport.send, None, sleep=sleeps.append)
    assert sleeps == [DEFAULT_RETRY_AFTER_SECONDS, MAX_RETRY_AFTER_SECONDS]


def test_exhaustion_is_typed_and_carries_every_request_id() -> None:
    transport = FakeTransport(*(busy(f"req-{i}") for i in range(CREATE_RETRY_ATTEMPTS)))
    sleeps: list[float] = []
    with pytest.raises(LaunchRetryExhausted) as caught:
        create_with_retry(transport.send, None, sleep=sleeps.append)
    error = caught.value
    assert len(transport.sent) == CREATE_RETRY_ATTEMPTS
    assert len(sleeps) == CREATE_RETRY_ATTEMPTS - 1
    assert error.request_ids == tuple(f"req-{i}" for i in range(CREATE_RETRY_ATTEMPTS))
    assert error.code == "launch_admission_busy"
    assert error.cause == "admission_lock_wait_exhausted:8000"
    assert "req-0" in str(error) and "req-3" in str(error)


@pytest.mark.parametrize(
    "error",
    [
        RefusedError("x", code="invalid_objective", status=422, retryable=False),
        ConflictError("x", code="launch_admission_busy", status=409),
        OutcomeUncertainError("uncertain", status=503),
    ],
)
def test_other_failures_are_never_retried(error: BaseException) -> None:
    transport = FakeTransport(error)
    with pytest.raises(type(error)):
        create_with_retry(transport.send, None, sleep=lambda _s: None)
    assert len(transport.sent) == 1


def test_async_path_retries_the_same_way() -> None:
    transport = FakeTransport(busy("r1", 0.5), "run-2")
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    result = asyncio.run(async_create_with_retry(transport.async_send, None, sleep=sleep))
    assert result == "run-2" and slept == [0.5] and len(transport.sent) == 2
