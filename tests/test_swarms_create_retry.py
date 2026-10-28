"""Run create retries a definite launch_admission_busy or owner_unavailable under fresh keys (SYN-4883, SYN-5091)."""

from __future__ import annotations

import asyncio

import pytest
from synth_ai.errors import ConflictError, OutcomeUncertainError, RefusedError
from synth_ai.options import MutationOptions
from synth_ai.types.refusal import Refusal
from synth_ai.sdk.swarms.create_retry import (
    CREATE_RETRY_ATTEMPTS,
    DEFAULT_RETRY_AFTER_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    RETRY_WAIT_BUDGET_SECONDS,
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


def owner_unavailable(request_id: str, retry_after: float | None = 1.0) -> RefusedError:
    """The backend's pre-acceptance owner readiness refusal (SYN-5091)."""
    error = RefusedError(
        "create refused: owner_unavailable",
        code="owner_unavailable",
        status=503,
        retry_after_seconds=retry_after,
        retryable=False,
        cause="port_readiness_owner_unavailable:blob_store=blob_write_unconfirmed:owner=agents",
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


def test_owner_unavailable_is_retried_under_a_fresh_key() -> None:
    """A transient owner readiness refusal (outcome=not_accepted) is resubmitted."""
    transport = FakeTransport(owner_unavailable("r1", 1.0), "run-3")
    sleeps: list[float] = []
    pinned = MutationOptions(idempotency_key="caller-key", request_id="r1")
    assert create_with_retry(transport.send, pinned, sleep=sleeps.append) == "run-3"
    assert sleeps == [1.0]
    retry = transport.sent[1]
    assert retry is not None
    assert retry.idempotency_key is None and retry.identity is None and retry.request_id is None


def test_owner_unavailable_exhaustion_keeps_its_code_and_cause() -> None:
    transport = FakeTransport(*(owner_unavailable(f"o-{i}") for i in range(CREATE_RETRY_ATTEMPTS)))
    with pytest.raises(LaunchRetryExhausted) as caught:
        create_with_retry(transport.send, None, sleep=lambda _s: None)
    assert len(transport.sent) == CREATE_RETRY_ATTEMPTS
    assert caught.value.code == "owner_unavailable"
    assert caught.value.cause.startswith("port_readiness_owner_unavailable:")
    assert caught.value.request_ids == tuple(f"o-{i}" for i in range(CREATE_RETRY_ATTEMPTS))


def test_retry_after_defaults_and_caps() -> None:
    transport = FakeTransport(busy("a", None), busy("b", 600.0), "ok")
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
        RefusedError("x", code="launch_preparation_refused", status=503, retryable=False),
        ConflictError("x", code="launch_admission_busy", status=409),
        ConflictError("x", code="owner_unavailable", status=409),
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


def backpressure(
    request_id: str,
    *,
    retryable: bool = True,
    retry_after: float | None = 15.0,
    code: str = "smr_control_plane_admission_backpressure",
) -> RefusedError:
    """SYN-5009: 429 per-project active-run cap with a retryable refusal document."""
    error = RefusedError(
        f"create refused: {code}",
        code=code,
        status=429,
        retry_after_seconds=retry_after,
        retryable=False,
        cause="project_active_runs_full",
        refusal=Refusal(
            schema_version="synth.refusal.v1",
            code=code,
            service="backend",
            stage="native_launch_preparation",
            retryable=retryable,
            in_doubt=False,
            operation_id="swarms.create",
            request_id=request_id,
            status=429,
            upstream=None,
            detail={
                "cause": "project_active_runs_full",
                "budget_scope": "project",
                "limit": "5",
                "held": "0",
                "pending": "5",
            },
        ),
    )
    error.client_request_id = request_id
    return error


def test_backpressure_429_retries_and_honors_retry_after_within_the_cap() -> None:
    transport = FakeTransport(backpressure("b1", retry_after=5.0), "run-4")
    sleeps: list[float] = []
    assert create_with_retry(transport.send, None, sleep=sleeps.append) == "run-4"
    assert sleeps == [5.0]
    assert len(transport.sent) == 2


def test_retry_after_15_is_waited_as_15() -> None:
    transport = FakeTransport(backpressure("b1", retry_after=15.0), "run-5")
    sleeps: list[float] = []
    create_with_retry(transport.send, None, sleep=sleeps.append)
    assert sleeps == [15.0]


def test_total_wait_budget_ends_the_retry() -> None:
    transport = FakeTransport(*(backpressure(f"w{i}", retry_after=50.0) for i in range(4)))
    sleeps: list[float] = []
    with pytest.raises(LaunchRetryExhausted):
        create_with_retry(transport.send, None, sleep=sleeps.append)
    assert sleeps == [50.0, 50.0]
    assert sum(sleeps) <= RETRY_WAIT_BUDGET_SECONDS


def test_older_server_code_without_flag_still_retries() -> None:
    transport = FakeTransport(backpressure("b1", retryable=False, retry_after=None), "run-6")
    sleeps: list[float] = []
    assert create_with_retry(transport.send, None, sleep=sleeps.append) == "run-6"
    assert sleeps == [DEFAULT_RETRY_AFTER_SECONDS]


def test_unknown_code_retries_only_when_the_document_says_retryable() -> None:
    transport = FakeTransport(backpressure("x1", code="future_capacity_full"), "run-7")
    assert create_with_retry(transport.send, None, sleep=lambda _s: None) == "run-7"
    refused = FakeTransport(backpressure("x2", code="future_capacity_full", retryable=False))
    with pytest.raises(RefusedError):
        create_with_retry(refused.send, None, sleep=lambda _s: None)
    assert len(refused.sent) == 1


def test_retryable_document_without_retry_after_is_not_retried() -> None:
    transport = FakeTransport(backpressure("x3", code="future_capacity_full", retry_after=None))
    with pytest.raises(RefusedError):
        create_with_retry(transport.send, None, sleep=lambda _s: None)
    assert len(transport.sent) == 1


def test_backpressure_exhaustion_surfaces_cause_and_cause_detail() -> None:
    transport = FakeTransport(*(backpressure(f"b-{i}") for i in range(CREATE_RETRY_ATTEMPTS)))
    with pytest.raises(LaunchRetryExhausted) as caught:
        create_with_retry(transport.send, None, sleep=lambda _s: None)
    error = caught.value
    assert error.code == "smr_control_plane_admission_backpressure"
    assert error.cause == "project_active_runs_full"
    assert error.cause_detail == "budget_scope=project held=0 pending=5 limit=5"
    assert "cause=project_active_runs_full" in str(error)
    assert "cause_detail=budget_scope=project held=0 pending=5 limit=5" in str(error)
