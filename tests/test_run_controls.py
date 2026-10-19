"""Keyed controls preserve command, scope and independent closure facts offline."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts import swarms as operations
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import MutationOptions
from synth_ai.sdk.swarms.run import AsyncSwarmRun, SwarmRun
from synth_ai.sdk.swarms.types.runs import RunSnapshot, SwarmEvent
from synth_ai.types.receipts import ControlReceipt

pytestmark = pytest.mark.unit
METADATA = {
    "contract": "swarms.v1",
    "request_id": "control-fixture",
    "operation_id": None,
    "observed_at": "2026-10-19T20:00:00Z",
}


def receipt(command=None, state="accepted"):
    value = {
        "metadata": METADATA,
        "operation_id": "control-1",
        "resource_id": "run-1",
        "state": state,
        "reason": None,
        "execution_closed": False,
        "accounting_final": False,
        "cleanup_complete": False,
    }
    if command is not None:
        value["command"] = command
    return value


@pytest.mark.parametrize("command", ["cancel", "pause", "resume"])
def test_sync_controls_send_one_keyed_command_and_preserve_pending_custody(command):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=receipt(command))

    with SynthClient(api_key="fixture-key", base_url="https://backend.test") as client:
        client._transport.client.close()
        client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
        client._transport._negotiate = lambda *args: None
        result = getattr(SwarmRun(client._transport, "run-1"), command)(
            mutation=MutationOptions(idempotency_key=f"key-{command}")
        )
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST" and request.url.path == f"/api/v1/swarms/runs/run-1/{command}"
    assert request.headers["Idempotency-Key"] == f"key-{command}"
    assert result.command == command and result.state == "accepted"
    assert (
        not result.execution_closed and not result.accounting_final and not result.cleanup_complete
    )


@pytest.mark.parametrize("command", ["cancel", "pause", "resume"])
def test_async_controls_use_the_same_command_contract(command):
    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200, headers={"X-Synth-Contract": "swarms.v1"}, json=receipt(command, "quiesced")
            )

        async with AsyncSynthClient(
            api_key="fixture-key", base_url="https://backend.test"
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(*args):
                return None

            client._transport._negotiate = negotiated
            result = await getattr(AsyncSwarmRun(client._transport, "run-1"), command)(
                mutation=MutationOptions(idempotency_key=f"key-{command}")
            )
            assert result.state == "quiesced" and result.command == command
            assert len(requests) == 1 and requests[0].headers["Idempotency-Key"] == f"key-{command}"

    asyncio.run(run())


def test_old_control_receipts_and_new_pause_projection_are_both_readable():
    assert ControlReceipt.model_validate_json(json.dumps(receipt())).command is None
    body = {
        "metadata": METADATA,
        "run_id": "run-1",
        "project_id": "project-1",
        "state": "pausing",
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": False,
        "execution_closed": False,
        "accounting_final": False,
        "cleanup_complete": False,
        "control": {"operation_id": "control-1", "command": "pause", "state": "applied"},
        "paused_at": "2026-10-19T20:00:00Z",
        "deadline_at": "2026-10-19T20:05:00Z",
        "resume_by": "2026-10-19T20:05:00Z",
    }
    snapshot = RunSnapshot.model_validate_json(json.dumps(body))
    assert snapshot.state == "pausing" and snapshot.control.command == "pause"
    assert snapshot.resume_by == snapshot.deadline_at
    for key in ("control", "paused_at", "resume_by", "deadline_at"):
        body.pop(key)
    assert RunSnapshot.model_validate_json(json.dumps(body)).control is None


@pytest.mark.parametrize("kind", ["paused", "resumed", "cancel_requested"])
def test_control_events_are_named_without_changing_usage_semantics(kind):
    event = SwarmEvent.model_validate_json(
        json.dumps({"event_id": "event-1", "run_id": "run-1", "sequence": 1, "kind": kind})
    )
    assert event.kind == kind and event.usage is None


def test_ops_tools_and_shipped_contract_preserve_control_scopes():
    contract = json.loads((Path(__file__).parents[1] / "contracts/swarms-v1.json").read_text())
    for command in ("pause", "resume"):
        op = getattr(operations, command.upper())
        assert op.id == f"swarms.{command}" and op.keyed and op.mutation and not op.may_charge
        assert op.required_scope == "swarms:control" and op in operations.ALL_OPERATIONS
        assert BY_NAME[f"swarms_{command}"].definition is op
        assert op.id in {row["id"] for row in contract["operations"]}
    assert "ControlReceipt" in contract["models"]
    with pytest.raises(ValidationError):
        ControlReceipt.model_validate_json(json.dumps(receipt("rerun")))
