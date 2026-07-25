"""A poll loop is its own retry: `swarms.wait` survives retryable failures.

A wait with time left on its deadline must not end because one poll lost a DNS
lookup. The distinction these tests pin matters to every caller that decides
whether to cancel durable work from a failed wait: a bare `TimeoutError` means
the Swarm was read and was not terminal, while a retryable error escaping the
wait means its state was never established.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from synth_ai.core.errors import (
    AuthorizationError,
    RetryDirective,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
    TransientServiceError,
)
from synth_ai.core.research.contracts.common import SwarmId
from synth_ai.core.research.swarms import AsyncSwarmsAPI, SwarmsAPI

SWARM_ID = SwarmId("11111111-1111-4111-8111-111111111111")


def _transport_failure() -> TransientServiceError:
    """The exact error the transport raises for a DNS or connection failure."""
    return TransientServiceError(
        0,
        f"/smr/runs/{SWARM_ID}",
        "network error (ConnectError)",
        failure=SynthFailure(
            code=SynthErrorCode("transport_error"),
            category=SynthErrorCategory.TRANSIENT_SERVICE,
            operation="retrieve_run",
            request_id=None,
            correlation_id=None,
            retry=RetryDirective(retryable=True),
            status=None,
            detail="GET transport failure",
        ),
    )


def _denial() -> AuthorizationError:
    return AuthorizationError(
        403,
        f"/smr/runs/{SWARM_ID}",
        "caller lacks authority for this run",
        failure=SynthFailure(
            code=SynthErrorCode("run_forbidden"),
            category=SynthErrorCategory.AUTHORIZATION,
            operation="retrieve_run",
            request_id=None,
            correlation_id=None,
            retry=RetryDirective(retryable=False),
            status=403,
            detail="forbidden",
        ),
    )


def _swarm(*, terminal: bool) -> Any:
    return SimpleNamespace(state=SimpleNamespace(is_terminal=terminal))


class _Outcomes:
    """Replay a scripted sequence of retrieve outcomes, exceptions included."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, _swarm_id: Any) -> Any:
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else _swarm(terminal=True)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def async_call(self, _swarm_id: Any) -> Any:
        return self(_swarm_id)


def _sync_api(outcomes: _Outcomes, monkeypatch: pytest.MonkeyPatch) -> SwarmsAPI:
    api = SwarmsAPI.__new__(SwarmsAPI)
    monkeypatch.setattr(api, "retrieve", outcomes, raising=False)
    return api


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(
        "synth_ai.core.research.swarms.time.sleep", lambda seconds: slept.append(seconds)
    )
    return slept


def test_wait_absorbs_a_transient_poll_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = _Outcomes([_transport_failure(), _swarm(terminal=False), _swarm(terminal=True)])
    slept = _no_sleep(monkeypatch)
    api = _sync_api(outcomes, monkeypatch)

    swarm = api.wait(SWARM_ID, timeout_seconds=60.0, poll_interval_seconds=2.0)

    assert swarm.state.is_terminal is True
    assert outcomes.calls == 3
    assert slept == [2.0, 2.0]


def test_wait_reraises_the_transient_failure_when_the_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Never reading the Swarm is not a verdict on it: the wait must not report a
    # plain timeout, which callers read as "read it, still running".
    outcomes = _Outcomes([_transport_failure() for _ in range(10)])
    _no_sleep(monkeypatch)
    api = _sync_api(outcomes, monkeypatch)

    clock = iter([0.0, 0.0, 1000.0, 1000.0])
    monkeypatch.setattr(
        "synth_ai.core.research.swarms.time.monotonic", lambda: next(clock)
    )

    with pytest.raises(TransientServiceError):
        api.wait(SWARM_ID, timeout_seconds=1.0, poll_interval_seconds=2.0)


def test_wait_reraises_a_non_retryable_denial_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _Outcomes([_denial(), _swarm(terminal=True)])
    _no_sleep(monkeypatch)
    api = _sync_api(outcomes, monkeypatch)

    with pytest.raises(AuthorizationError):
        api.wait(SWARM_ID, timeout_seconds=60.0, poll_interval_seconds=2.0)
    assert outcomes.calls == 1


def test_wait_still_times_out_on_an_authoritative_nonterminal_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _Outcomes([_swarm(terminal=False) for _ in range(10)])
    _no_sleep(monkeypatch)
    api = _sync_api(outcomes, monkeypatch)

    clock = iter([0.0, 1000.0])
    monkeypatch.setattr(
        "synth_ai.core.research.swarms.time.monotonic", lambda: next(clock)
    )

    with pytest.raises(TimeoutError):
        api.wait(SWARM_ID, timeout_seconds=1.0, poll_interval_seconds=2.0)


def test_async_wait_absorbs_a_transient_poll_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _Outcomes([_transport_failure(), _swarm(terminal=True)])
    api = AsyncSwarmsAPI.__new__(AsyncSwarmsAPI)
    monkeypatch.setattr(api, "retrieve", outcomes.async_call, raising=False)

    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("synth_ai.core.research.swarms.asyncio.sleep", _sleep)

    swarm = asyncio.run(
        api.wait(SWARM_ID, timeout_seconds=60.0, poll_interval_seconds=2.0)
    )

    assert swarm.state.is_terminal is True
    assert outcomes.calls == 2
    assert slept == [2.0]
