"""SYN-3992: ordinary SDK and MCP preserve declared scientific outcomes.

All transports are in-process. No providers, sockets or manufactured execution
custody. See Forge docs/contracts.md and backend forge_scientific_delivery.md.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from synth_ai.mcp.research.tools.scientific_records import build_scientific_record_tools
from synth_ai.sdk.research.contracts.forge.contracts import contract_digest
from synth_ai.sdk.research.contracts.forge.operations import PublicWrite
from synth_ai.sdk.research.contracts.forge.records import Result
from synth_ai.sdk.research.scientific_records import (
    AsyncScientificRecordsAPI,
    ExecutionWriteRequest,
    ScientificRecordsAPI,
)

ORG = "00000000-0000-4000-8000-00000000000a"
PROJECT = "00000000-0000-4000-8000-00000000000b"


def payload(outcome, value):
    def reference(kind):
        return {
            "authority": "fixture",
            "kind": kind,
            "record_id": kind,
            "revision": "1",
            "digest_sha256": "a" * 64,
        }

    return {
        "kind": "result",
        "trial": reference("trial"),
        "outcome": outcome,
        "measurements": [{"name": "reward", "value": value, "unit": "score"}],
        "evaluator": reference("evaluator"),
        "evidence": [reference("artifact")],
    }


def request(outcome, value):
    values = {
        "operation_id": "same-intent",
        "record_id": "result",
        "expected_revision": 0,
        "payload": Result.model_validate(payload(outcome, value)).model_dump(mode="json"),
    }
    write = PublicWrite(**values, request_digest_sha256=contract_digest(values))
    return ExecutionWriteRequest(organization_id=ORG, admission_id="adm_" + "0" * 26, write=write)


def receipt(write):
    return {
        "schema_version": "forge.receipt.v1",
        "scope": {"organization_id": ORG, "project_id": PROJECT},
        "operation_id": write.operation_id,
        "payload_digest_sha256": "c" * 64,
        "reference": {
            "authority": "forge",
            "kind": "result",
            "record_id": write.record_id,
            "revision": "1",
            "digest_sha256": contract_digest(write.payload),
        },
        "cursor": 1,
        "recorded_at": "2026-10-07T00:00:00Z",
    }


@pytest.mark.parametrize(
    "outcome,value",
    [
        ("null", None),
        ("negative", -0.5),
        ("inconclusive", None),
        ("failed", None),
        ("interrupted", None),
        ("rejected", None),
        ("measured", 0.0),
    ],
)
def test_real_mcp_execution_result_preserves_outcome__syn3992(outcome, value):
    selected = request(outcome, value)
    captured = []

    def send(method, path, **arguments):
        captured.append((method, path, arguments))
        return receipt(selected.write)

    @contextmanager
    def client_factory(_):
        yield SimpleNamespace(records=ScientificRecordsAPI(SimpleNamespace(request_json=send)))

    tools = build_scientific_record_tools(client_factory)
    tool = next(item for item in tools if item.name == "research_record_execution_result")
    answer = tool.handler({"project_id": PROJECT, **selected.model_dump(mode="json")})
    assert answer["reference"]["digest_sha256"] == contract_digest(selected.write.payload)
    body = captured[0][2]["json_body"]
    assert captured[0][0] == "POST" and captured[0][1].endswith("/execution-operations")
    assert body["write"]["payload"]["outcome"] == outcome
    assert body["write"]["payload"]["measurements"][0]["value"] == value
    assert body["admission_id"] == selected.admission_id and "producer" not in body["write"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,value", [("null", None), ("negative", -0.5), ("inconclusive", None)]
)
async def test_async_execution_result_keeps_exact_nullable_value__syn3992(outcome, value):
    selected = request(outcome, value)
    captured = []

    async def send(method, path, **arguments):
        captured.append(arguments)
        return receipt(selected.write)

    result = await AsyncScientificRecordsAPI(SimpleNamespace(request_json=send)).execution_write(
        PROJECT, selected
    )
    assert result.reference.digest_sha256 == contract_digest(selected.write.payload)
    assert captured[0]["json_body"]["write"]["payload"]["outcome"] == outcome
    assert captured[0]["json_body"]["write"]["payload"]["measurements"][0]["value"] == value


def test_mcp_all_null_measured_intent_refuses_before_client__syn3992():
    def client_factory(_):
        raise AssertionError("Invalid scientific intent reached client/effects")

    tools = build_scientific_record_tools(client_factory)
    tool = next(item for item in tools if item.name == "research_record_result")
    with pytest.raises(ValidationError, match="numeric measurement"):
        tool.handler(
            {
                "project_id": PROJECT,
                "record_id": "result",
                "operation_id": "same-intent",
                "payload": payload("measured", None),
            }
        )
