"""Core-ergonomics hero flows: launch_and_wait, retry, expressibility, terminal helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models.run_state import RunState
from synth_ai.managed_research.sdk.client import _build_project_run_payload
from synth_ai.managed_research.sdk.runs import RunHandle
from synth_ai.research.swarms import (
    ResearchSwarmHandle,
    ResearchSwarmsAPI,
    SwarmLaunchBackpressureError,
    SwarmPreflightBlockedError,
    SwarmResult,
    SwarmRetryResult,
    classify_event_kind,
    swarm_state_is_terminal,
)

_USAGE_SENTINEL = {"total_tokens": 123}
_COST_SENTINEL = {"total_usd": 4.56}
_WORK_PRODUCTS_SENTINEL = [{"work_product_id": "wp-1", "kind": "report"}]


class _FakeRunsNamespace:
    def __init__(self, client: FakeSessionClient) -> None:
        self._client = client

    def launch_preflight(
        self,
        project_id: str | None = None,
        *,
        project: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._client.preflight_calls.append({"project_id": project_id, **kwargs})
        return dict(self._client.preflight_payload)

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: Any = None,
        **kwargs: Any,
    ) -> RunHandle:
        self._client.start_calls.append({"objective": objective, **kwargs})
        self._client.maybe_raise_launch_error()
        return RunHandle(self._client, project_id or "proj-1", self._client.launched_run_id)

    def trigger(
        self,
        project_id: str | None = None,
        *,
        project: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._client.trigger_calls.append({"project_id": project_id, **kwargs})
        self._client.maybe_raise_launch_error()
        return {
            "run_id": self._client.launched_run_id,
            "project_id": project_id or "proj-1",
            "public_state": "queued",
        }


class _FakeWorkProductsNamespace:
    def __init__(self, client: FakeSessionClient) -> None:
        self._client = client

    def list_for_run(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        if self._client.readouts_raise:
            raise SmrApiError("work products unavailable", status_code=500)
        return list(_WORK_PRODUCTS_SENTINEL)


class FakeSessionClient:
    """Duck-typed ManagedResearchClient standing in for the network client."""

    def __init__(self) -> None:
        self.runs = _FakeRunsNamespace(self)
        self.work_products = _FakeWorkProductsNamespace(self)
        self.preflight_payload: dict[str, Any] = {"clear_to_trigger": True, "blockers": []}
        self.final_state = "done"
        self.launched_run_id = "swarm-1"
        self.launch_errors: list[SmrApiError] = []
        self.readouts_raise = False
        self.preflight_calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.trigger_calls: list[dict[str, Any]] = []
        self.branch_calls: list[dict[str, Any]] = []

    def maybe_raise_launch_error(self) -> None:
        if self.launch_errors:
            raise self.launch_errors.pop(0)

    def run(self, project_id: str, run_id: str) -> RunHandle:
        return RunHandle(self, project_id, run_id)

    def get_run_contract(self, project_id: str, run_id: str) -> Any:
        return SimpleNamespace(
            terminal=swarm_state_is_terminal(self.final_state),
            public_state=SimpleNamespace(value=self.final_state),
        )

    def get_project_run(self, project_id: str, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "project_id": project_id,
            "public_state": self.final_state,
        }

    def get_run_usage(self, run_id: str) -> dict[str, Any]:
        if self.readouts_raise:
            raise SmrApiError("usage unavailable", status_code=500)
        return dict(_USAGE_SENTINEL)

    def get_run_cost_summary(self, run_id: str) -> dict[str, Any]:
        if self.readouts_raise:
            raise SmrApiError("cost unavailable", status_code=500)
        return dict(_COST_SENTINEL)

    def branch_run_from_checkpoint(self, run_id: str, **kwargs: Any) -> Any:
        self.branch_calls.append({"run_id": run_id, **kwargs})
        return SimpleNamespace(child_run_id="swarm-2", parent_run_id=run_id)


def _api(client: FakeSessionClient) -> ResearchSwarmsAPI:
    return ResearchSwarmsAPI(client)  # type: ignore[arg-type]


def _handle(client: FakeSessionClient, run_id: str = "swarm-1") -> ResearchSwarmHandle:
    return ResearchSwarmHandle(RunHandle(client, "proj-1", run_id))


# --- launch_and_wait -----------------------------------------------------------


def test_launch_and_wait_success_fetches_typed_readouts() -> None:
    client = FakeSessionClient()
    result = _api(client).launch_and_wait(
        "proj-1",
        objective="Audit the repo",
        timeout=60.0,
    )
    assert isinstance(result, SwarmResult)
    assert result.swarm_id == "swarm-1"
    assert result.project_id == "proj-1"
    assert result.status == "done"
    assert result.is_success is True
    assert result.is_terminal is True
    assert result.usage == _USAGE_SENTINEL
    assert result.cost == _COST_SENTINEL
    assert result.work_products == _WORK_PRODUCTS_SENTINEL
    assert isinstance(result.handle, ResearchSwarmHandle)
    assert result.swarm.public_state is RunState.DONE
    assert len(client.preflight_calls) == 1


def test_launch_and_wait_timeout_is_required_and_positive() -> None:
    client = FakeSessionClient()
    with pytest.raises(TypeError):
        _api(client).launch_and_wait("proj-1", objective="x")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="timeout"):
        _api(client).launch_and_wait("proj-1", objective="x", timeout=0.0)


def test_launch_and_wait_preflight_blocked_raises_typed_error() -> None:
    client = FakeSessionClient()
    client.preflight_payload = {
        "clear_to_trigger": False,
        "blockers": [{"blocker": "missing_repo"}, {"blocker": "no_budget"}],
    }
    with pytest.raises(SwarmPreflightBlockedError) as excinfo:
        _api(client).launch_and_wait("proj-1", objective="x", timeout=5.0)
    assert excinfo.value.blockers == [
        {"blocker": "missing_repo"},
        {"blocker": "no_budget"},
    ]
    assert "missing_repo" in str(excinfo.value)
    assert client.start_calls == []


def test_launch_and_wait_retries_backpressure_then_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "synth_ai.research.swarms.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    client = FakeSessionClient()
    client.launch_errors = [SmrApiError("edge 503", status_code=503) for _ in range(5)]
    with pytest.raises(SwarmLaunchBackpressureError) as excinfo:
        _api(client).launch_and_wait("proj-1", objective="x", timeout=5.0)
    assert excinfo.value.attempts == 5
    assert excinfo.value.last_error.status_code == 503
    assert "SmrApiError" in str(excinfo.value)
    assert len(sleeps) == 4
    assert all(seconds <= 30.0 for seconds in sleeps)


def test_launch_and_wait_retries_transient_backpressure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("synth_ai.research.swarms.time.sleep", lambda seconds: None)
    client = FakeSessionClient()
    client.launch_errors = [SmrApiError("edge 502", status_code=502)]
    result = _api(client).launch_and_wait("proj-1", objective="x", timeout=5.0)
    assert result.is_success is True
    assert len(client.start_calls) == 2


def test_launch_and_wait_non_retryable_error_raises_immediately() -> None:
    client = FakeSessionClient()
    client.launch_errors = [SmrApiError("bad request", status_code=422)]
    with pytest.raises(SmrApiError) as excinfo:
        _api(client).launch_and_wait("proj-1", objective="x", timeout=5.0)
    assert not isinstance(excinfo.value, SwarmLaunchBackpressureError)
    assert len(client.start_calls) == 1


def test_launch_and_wait_failed_swarm_uses_best_effort_readouts() -> None:
    client = FakeSessionClient()
    client.final_state = "failed"
    client.readouts_raise = True
    result = _api(client).launch_and_wait("proj-1", objective="x", timeout=5.0)
    assert result.is_success is False
    assert result.status == "failed"
    assert result.is_terminal is True
    assert result.usage is None
    assert result.cost is None
    assert result.work_products == []


def test_launch_and_wait_configured_path_uses_trigger() -> None:
    client = FakeSessionClient()
    result = _api(client).launch_and_wait("proj-1", timeout=5.0)
    assert result.is_success is True
    assert len(client.trigger_calls) == 1
    assert client.start_calls == []


# --- launch expressibility -----------------------------------------------------


def test_build_project_run_payload_passes_execution_target_and_local_launch() -> None:
    payload = _build_project_run_payload(
        local_execution={"slot": "slot-2"},
        execution_profile={"profile": "local-dev"},
        execution_target={"kind": "slot", "slot_id": "slot-2"},
    )
    assert payload["local_execution"] == {"slot": "slot-2"}
    assert payload["execution_profile"] == {"profile": "local-dev"}
    assert payload["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}


def test_run_launch_request_accepts_execution_target() -> None:
    from synth_ai.managed_research.models.run_launch import RunLaunchRequest

    request = RunLaunchRequest(
        runbook_preset="quick",
        execution_target={"kind": "slot", "slot_id": "slot-2"},
        local_execution={"slot": "slot-2"},
    )
    kwargs = request.to_client_kwargs()
    assert kwargs["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}
    wire = request.to_wire()
    assert wire["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}
    assert wire["local_execution"] == {"slot": "slot-2"}


def test_facade_create_configured_passes_execution_target_through() -> None:
    client = FakeSessionClient()
    handle = _api(client).create_configured(
        "proj-1",
        execution_target={"kind": "slot", "slot_id": "slot-2"},
        local_execution={"slot": "slot-2"},
    )
    assert isinstance(handle, ResearchSwarmHandle)
    (call,) = client.trigger_calls
    assert call["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}
    assert call["local_execution"] == {"slot": "slot-2"}


def test_launch_and_wait_passes_execution_target_to_preflight_and_launch() -> None:
    client = FakeSessionClient()
    _api(client).launch_and_wait(
        "proj-1",
        timeout=5.0,
        execution_target={"kind": "slot", "slot_id": "slot-2"},
    )
    (preflight_call,) = client.preflight_calls
    (trigger_call,) = client.trigger_calls
    assert preflight_call["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}
    assert trigger_call["execution_target"] == {"kind": "slot", "slot_id": "slot-2"}


# --- retry ---------------------------------------------------------------------


def test_retry_rejects_non_terminal_swarm_with_state_name() -> None:
    client = FakeSessionClient()
    client.final_state = "running"
    with pytest.raises(ValueError, match="running"):
        _handle(client).retry()


def test_retry_fresh_relaunches_configured_swarm() -> None:
    client = FakeSessionClient()
    client.final_state = "failed"
    client.launched_run_id = "swarm-2"
    result = _handle(client).retry(reason="flaky infra")
    assert isinstance(result, SwarmRetryResult)
    assert result.source_swarm_id == "swarm-1"
    assert result.new_swarm_id == "swarm-2"
    assert result.mode == "fresh"
    assert result.reason == "flaky infra"
    assert result.checkpoint_id is None
    assert result.handle.swarm_id == "swarm-2"
    assert len(client.trigger_calls) == 1


def test_retry_from_checkpoint_branches_from_checkpoint() -> None:
    client = FakeSessionClient()
    client.final_state = "failed"
    result = _handle(client).retry(
        "from_checkpoint",
        reason="resume after fix",
        checkpoint_id="ckpt-9",
    )
    assert result.mode == "from_checkpoint"
    assert result.new_swarm_id == "swarm-2"
    assert result.checkpoint_id == "ckpt-9"
    assert result.handle.swarm_id == "swarm-2"
    (call,) = client.branch_calls
    assert call["checkpoint_id"] == "ckpt-9"
    assert call["reason"] == "resume after fix"


def test_retry_argument_gating() -> None:
    client = FakeSessionClient()
    client.final_state = "done"
    with pytest.raises(ValueError, match="checkpoint_id"):
        _handle(client).retry("from_checkpoint")
    with pytest.raises(ValueError, match="checkpoint_id"):
        _handle(client).retry("fresh", checkpoint_id="ckpt-9")
    with pytest.raises(ValueError, match="mode"):
        _handle(client).retry("sideways")  # type: ignore[arg-type]


# --- observe polish ------------------------------------------------------------


def test_wait_until_terminal_returns_final_state() -> None:
    client = FakeSessionClient()
    swarm = _handle(client).wait_until_terminal(timeout=5.0)
    assert swarm.public_state is RunState.DONE


def test_handle_is_terminal_uses_contract_authority() -> None:
    client = FakeSessionClient()
    assert _handle(client).is_terminal() is True
    client.final_state = "running"
    assert _handle(client).is_terminal() is False


def test_swarm_state_is_terminal_reuses_run_state_source() -> None:
    for state in RunState:
        assert swarm_state_is_terminal(state.value) == state.is_terminal
    assert swarm_state_is_terminal("some_future_state") is False
    assert swarm_state_is_terminal("") is False


@pytest.mark.parametrize(
    ("kind", "category"),
    [
        ("tool.call.started", "tool"),
        ("tool.call.completed", "tool"),
        ("message.delta", "message"),
        ("operator.message.sent", "message"),
        ("reasoning.summary", "message"),
        ("turn.completed", "turn"),
        ("token.usage", "usage"),
        ("run.state.changed", "status"),
        ("task.state.changed", "status"),
        ("heartbeat", "status"),
        ("some.future.kind", "other"),
        ("", "other"),
    ],
)
def test_classify_event_kind(kind: str, category: str) -> None:
    assert classify_event_kind(kind) == category
