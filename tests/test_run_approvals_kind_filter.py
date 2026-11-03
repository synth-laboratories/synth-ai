"""SYN-4650: approvals(kind=...) sends ?kind= and parses recorded gate rows offline."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ContractError
from synth_ai.sdk.swarms.run import AsyncSwarmRun, SwarmRun
from synth_ai.sdk.swarms.types.approvals import GateApproval

pytestmark = pytest.mark.unit
PACKAGE = Path(__file__).parent / "fixtures" / "swarms-v1-approval-gate"
METADATA = {
    "contract": "swarms.v1",
    "request_id": "gate-filter-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}


def gate_row(name: str = "gate.release.accepted.json") -> dict:
    # Recorded backend bytes (synth.swarms-approval-gate.v1), not hand-built rows.
    return {"metadata": METADATA, **json.loads((PACKAGE / name).read_text())}


RUN_ID = gate_row()["run_id"]
TOOL_CALL = {
    "metadata": METADATA,
    "approval_id": "approval-1",
    "run_id": RUN_ID,
    "revision": 3,
    "state": "awaiting",
    "kind": "tool_call",
    "question": "Allow tool git_push?",
    "tool": "git_push",
    "capability": "git.push.remote",
    "argument_digest": "sha256:" + "a" * 64,
    "actor_id": "actor-1",
    "attempt_id": "attempt-1",
    "parked_at": "2026-10-29T11:59:00Z",
    "deadline_at": None,
    "on_timeout": None,
    "decidable": True,
}


def listing(*items: dict) -> dict:
    return {"metadata": METADATA, "run_id": RUN_ID, "items": list(items)}


def sync_client(respond):
    client = SynthClient(api_key="fixture-key", base_url="https://backend.test")
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    client._transport._negotiate = lambda *args: None
    return client


def serve(body: dict, requests: list):
    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=body)

    return respond


def test_gate_filter_is_on_the_wire_and_parses_the_recorded_gate_row():
    requests = []
    with sync_client(serve(listing(gate_row()), requests)) as client:
        result = SwarmRun(client._transport, RUN_ID).approvals(kind="gate")
    (request,) = requests
    assert request.url.path == f"/api/v1/swarms/runs/{RUN_ID}/approvals"
    assert dict(request.url.params) == {"kind": "gate"}
    (gate,) = result.items
    assert isinstance(gate, GateApproval)
    assert (gate.approval_id, gate.run_id, gate.gate_kind, gate.state) == (
        "00000000-0000-0000-0000-00000000001f",
        RUN_ID,
        "release",
        "awaiting",
    )
    assert (gate.revision, gate.decidable, gate.target_task_id) == (
        1,
        True,
        "00000000-0000-4000-8000-0000000000f1",
    )


def test_no_kind_sends_no_query():
    requests = []
    with sync_client(serve(listing(TOOL_CALL, gate_row()), requests)) as client:
        result = SwarmRun(client._transport, RUN_ID).approvals()
    assert dict(requests[0].url.params) == {}
    assert [item.kind for item in result.items] == ["tool_call", "gate"]


def test_an_item_of_another_kind_under_a_filter_is_a_contract_error():
    with (
        sync_client(serve(listing(TOOL_CALL, gate_row()), [])) as client,
        pytest.raises(ContractError),
    ):
        SwarmRun(client._transport, RUN_ID).approvals(kind="gate")


def test_an_unknown_kind_is_refused_before_any_request():
    requests = []
    with (
        sync_client(serve(listing(), requests)) as client,
        pytest.raises(ContractError) as refused,
    ):
        SwarmRun(client._transport, RUN_ID).approvals(kind="question")  # type: ignore[arg-type]
    assert refused.value.code == "approval_kind_invalid"
    assert requests == []


@pytest.mark.parametrize(
    "name", ["gate.unknown-gate-kind.refused.json", "gate.revision-2.refused.json"]
)
def test_a_recorded_refused_gate_row_is_a_contract_error(name):
    with (
        sync_client(serve(listing(gate_row(name)), [])) as client,
        pytest.raises(ContractError),
    ):
        SwarmRun(client._transport, RUN_ID).approvals(kind="gate")


def test_async_gate_filter_uses_the_same_contract():
    async def run():
        requests = []
        async with AsyncSynthClient(
            api_key="fixture-key", base_url="https://backend.test"
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(
                transport=httpx.MockTransport(serve(listing(gate_row()), requests))
            )

            async def negotiated(*args):
                return None

            client._transport._negotiate = negotiated
            result = await AsyncSwarmRun(client._transport, RUN_ID).approvals(kind="gate")
        return requests, result

    requests, result = asyncio.run(run())
    assert dict(requests[0].url.params) == {"kind": "gate"}
    assert isinstance(result.items[0], GateApproval)
