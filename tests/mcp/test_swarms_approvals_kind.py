"""MCP ``swarms_approvals`` takes the SDK's closed ``kind`` and puts it on the wire."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import AsyncSynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.contracts import ApprovalListArguments
from synth_ai.sdk.swarms.run import AsyncSwarmRun
from synth_ai.sdk.swarms.types.approvals import ApprovalKind, GateApproval
from synth_ai.types.base import Model

pytestmark = pytest.mark.unit
PACKAGE = Path(__file__).parents[1] / "fixtures" / "swarms-v1-approval-gate"
METADATA = {
    "contract": "swarms.v1",
    "request_id": "mcp-gate-filter-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}
TOOL = BY_NAME["swarms_approvals"]


def recorded_gate() -> dict:
    # Recorded backend bytes (synth.swarms-approval-gate.v1), not a hand-built row.
    return {
        "metadata": METADATA,
        **json.loads((PACKAGE / "gate.release.accepted.json").read_text()),
    }


RUN_ID = recorded_gate()["run_id"]


def call(arguments: dict, body: dict) -> tuple[list[httpx.Request], object]:
    async def run():
        requests: list[httpx.Request] = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=body)

        async with AsyncSynthClient(
            api_key="fixture-key", base_url="https://backend.test"
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(*args):
                return None

            client._transport._negotiate = negotiated

            async def get(run_id):
                return AsyncSwarmRun(client._transport, run_id)

            facade = SimpleNamespace(swarms=SimpleNamespace(get=get))
            result = await TOOL.execute(facade, TOOL.request.model_validate(arguments))
        return requests, result

    return asyncio.run(run())


def listing(*items: dict) -> dict:
    return {"metadata": METADATA, "run_id": RUN_ID, "items": list(items)}


def test_the_tool_schema_documents_the_closed_kind_values():
    assert TOOL.request is ApprovalListArguments
    schema = TOOL.request.model_json_schema()
    kind = schema["properties"]["kind"]
    (closed,) = [branch for branch in kind["anyOf"] if branch.get("type") == "string"]
    assert closed["enum"] == list(get_args(ApprovalKind)) == ["tool_call", "wait", "gate"]
    assert "gate" in kind["description"]
    assert schema["required"] == ["run_id"]
    assert "kind=tool_call|wait|gate" in TOOL.description


def test_kind_gate_is_forwarded_and_parses_the_recorded_gate_row():
    requests, result = call({"run_id": RUN_ID, "kind": "gate"}, listing(recorded_gate()))
    (request,) = requests
    assert request.url.path == f"/api/v1/swarms/runs/{RUN_ID}/approvals"
    assert dict(request.url.params) == {"kind": "gate"}
    (gate,) = result.items
    assert isinstance(gate, GateApproval)


def test_no_kind_sends_no_query():
    requests, result = call({"run_id": RUN_ID}, listing(recorded_gate()))
    assert dict(requests[0].url.params) == {}
    assert isinstance(result.items[0], GateApproval)


def test_a_row_of_another_kind_under_the_filter_is_a_contract_error():
    with pytest.raises(ContractError):
        call({"run_id": RUN_ID, "kind": "wait"}, listing(recorded_gate()))


@pytest.mark.parametrize("kind", ["GATE", "approval", "", 1])
def test_an_unknown_kind_is_refused_before_any_request(kind):
    with pytest.raises(ValidationError):
        ApprovalListArguments.model_validate({"run_id": RUN_ID, "kind": kind})


def test_extra_arguments_stay_closed():
    assert issubclass(ApprovalListArguments, Model)
    with pytest.raises(ValidationError):
        ApprovalListArguments.model_validate({"run_id": RUN_ID, "state": "awaiting"})
