"""Control recovery never repeats POST; state observations have finite budgets."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from synth_ai.errors import OutcomeUncertainError, RequestTimeoutError, ResultNotReadyError
from synth_ai.options import ObservationOptions
from synth_ai.sdk.swarms.run import AsyncSwarmRun, SwarmRun
from synth_ai.sdk.swarms.types.runs import RunSnapshot
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import ControlReceipt, MutationResolution
from tests.test_run_controls import METADATA, receipt

pytestmark = pytest.mark.unit


class Transport:
    request_timeout_seconds = 30

    def __init__(self, command, *, resolution="accepted", mismatch=False, identity=True):
        self.command, self.resolution, self.mismatch = command, resolution, mismatch
        self.calls = []
        self.identity = MutationIdentity.create(
            "swarms." + command, b"", f"/api/v1/swarms/runs/run-1/{command}", key="same-key"
        )
        self.error = OutcomeUncertainError(
            "lost reply", identity=self.identity if identity else None
        )

    def request(self, operation, model, *, validate=None, options=None, **arguments):
        self.calls.append((operation.id, arguments, options))
        if operation.mutation:
            raise self.error
        if operation.id == "swarms.resolve":
            selected = (
                self.identity.model_copy(update={"key": "other"})
                if self.mismatch
                else self.identity
            )
            result = MutationResolution.model_validate_json(
                json.dumps(
                    {
                        "metadata": {**METADATA, "operation_id": "control-1"},
                        "identity": selected.model_dump(mode="json"),
                        "state": self.resolution,
                        "resource": {"kind": "control", "resource_id": "underlying-owner-id"}
                        if self.resolution == "accepted"
                        else None,
                    }
                )
            )
        else:
            result = ControlReceipt.model_validate_json(
                json.dumps(receipt(self.command, "applied"))
            )
        return validate(result) if validate else result


class AsyncTransport(Transport):
    async def request(self, *args, **kwargs):
        return super().request(*args, **kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("command", ["cancel", "pause", "resume"])
@pytest.mark.parametrize("resolution", ["accepted", "pending", "not_accepted"])
def test_uncertain_control_resolves_public_custody_then_reads_exact_receipt(
    async_mode, command, resolution
):
    transport = (AsyncTransport if async_mode else Transport)(command, resolution=resolution)
    run = (AsyncSwarmRun if async_mode else SwarmRun)(transport, "run-1")
    result = asyncio.run(getattr(run, command)()) if async_mode else getattr(run, command)()
    assert result.command == command
    assert [row[0] for row in transport.calls] == [
        "swarms." + command,
        "swarms.resolve",
        "swarms.control",
    ]
    assert transport.calls[1][1]["query"]["key"] == "same-key"
    assert transport.calls[2][1]["path"].endswith("/controls/control-1")
    assert all(row[2].read_retry_count == 0 for row in transport.calls[1:])
    assert 0 < transport.calls[2][2].timeout_seconds <= transport.calls[1][2].timeout_seconds <= 30


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("fault", ["unknown", "identity_missing", "mismatch"])
def test_incomplete_recovery_preserves_original_uncertainty_and_does_not_resubmit(
    async_mode, fault
):
    transport = (AsyncTransport if async_mode else Transport)(
        "pause",
        resolution="unavailable" if fault == "unknown" else "accepted",
        identity=fault != "identity_missing",
        mismatch=fault == "mismatch",
    )
    run = (AsyncSwarmRun if async_mode else SwarmRun)(transport, "run-1")
    with pytest.raises(OutcomeUncertainError) as caught:
        asyncio.run(run.pause()) if async_mode else run.pause()
    assert caught.value is transport.error
    assert sum(row[0] == "swarms.pause" for row in transport.calls) == 1
    assert len(transport.calls) == (1 if fault == "identity_missing" else 2)
    if fault == "mismatch":
        assert caught.value.__cause__ is not None


def snapshot(state):
    return RunSnapshot.model_validate_json(
        json.dumps(
            {
                "metadata": METADATA,
                "run_id": "run-1",
                "project_id": "project-1",
                "state": state,
                "model": "gpt-6-luna",
                "provider": "openai",
                "result_available": False,
                "execution_closed": state == "cancelled",
                "accounting_final": False,
                "cleanup_complete": False,
            }
        )
    )


OBSERVATION = ObservationOptions(
    timeout_seconds=1,
    idle_timeout_seconds=1,
    reconnect_deadline_seconds=1,
    retry_count_max=0,
    maximum_events=2,
    poll_interval_seconds=0.1,
)


@pytest.mark.parametrize("async_mode", [False, True])
def test_pause_and_target_state_are_observed_without_result_polling_or_mutation(
    monkeypatch, async_mode
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("synth_ai.sdk.swarms.run.time.monotonic", lambda: clock.now)
    monkeypatch.setattr(
        "synth_ai.sdk.swarms.run.time.sleep",
        lambda amount: setattr(clock, "now", clock.now + amount),
    )

    async def sleep(amount):
        clock.now += amount

    monkeypatch.setattr("synth_ai.sdk.swarms.run.asyncio.sleep", sleep)

    class Reads:
        def __init__(self, states):
            self.states = iter(states)
            self.calls = []

        def request(self, operation, model, **kwargs):
            self.calls.append(operation.id)
            assert operation.id == "swarms.get"
            return snapshot(next(self.states))

    class AsyncReads(Reads):
        async def request(self, *args, **kwargs):
            return super().request(*args, **kwargs)

    def observe(states, method, *args):
        transport = (AsyncReads if async_mode else Reads)(states)
        run = (AsyncSwarmRun if async_mode else SwarmRun)(transport, "run-1")
        value = getattr(run, method)(*args, observation=OBSERVATION)
        return (asyncio.run(value) if async_mode else value), transport

    value, transport = observe(["pausing", "paused"], "wait_for_state", "paused")
    assert value.state == "paused" and len(transport.calls) == 2
    with pytest.raises(ResultNotReadyError, match="run_paused"):
        observe(["paused"], "wait")
    with pytest.raises(ResultNotReadyError, match="run_state_unreachable"):
        observe(["cancelled"], "wait_for_state", "running")
    with pytest.raises(RequestTimeoutError):
        observe(["running", "running"], "wait_for_state", "paused")


@pytest.mark.parametrize("async_mode", [False, True])
def test_expired_mutation_budget_does_not_start_a_reconciliation_read(monkeypatch, async_mode):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("synth_ai.sdk.swarms.run.time.monotonic", lambda: clock.now)

    class Expired(Transport):
        def request(self, *args, **kwargs):
            clock.now = 31
            return super().request(*args, **kwargs)

    class AsyncExpired(Expired):
        async def request(self, *args, **kwargs):
            return super().request(*args, **kwargs)

    transport = (AsyncExpired if async_mode else Expired)("pause")
    run = (AsyncSwarmRun if async_mode else SwarmRun)(transport, "run-1")
    with pytest.raises(OutcomeUncertainError):
        asyncio.run(run.pause()) if async_mode else run.pause()
    assert len(transport.calls) == 1
