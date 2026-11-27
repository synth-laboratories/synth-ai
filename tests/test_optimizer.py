"""Optimizer contract 1.28 follows the integrated Core schema, without inferred pins."""

import json

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from test_cli import example

from synth_ai import AsyncSynthClient, OptimizerPin, SynthClient
from synth_ai._generated.models import FamilyList, LaunchRequest, RunSnapshot
from synth_ai.cli import parse_optimizer, parser
from synth_ai.mcp.server import call_tool

RECEIPT = {
    "topology": "swarms_v2",
    "version": 2,
    "params": {},
    "reason": "environment_default",
    "definition_digest": "sha256:" + "a" * 64,
}


async def test_sdk_sends_explicit_auto_and_reads_immutable_resolution(contract):
    requests = []
    _, snapshot = example(contract, "getRun")
    snapshot["optimizer"] = RECEIPT

    def handle(request):
        requests.append(request)
        return httpx.Response(202, json=snapshot)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            run = await client.swarms.launch("task", budget="1.00", optimizer="auto")
    assert json.loads(requests[0].content)["optimizer"] == "auto"
    assert run.optimizer is not None
    assert run.optimizer.topology == "swarms_v2"
    assert run.optimizer.definition_digest == RECEIPT["definition_digest"]
    assert RunSnapshot.model_validate_json(run.model_dump_json()).optimizer == run.optimizer


def test_typed_pin_is_accepted_and_round_trips():
    request = {
        "message": "task",
        "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
        "optimizer": {"topology": "swarms_v2", "version": 2, "params": {}},
    }
    model = LaunchRequest.model_validate(request)
    assert model.model_dump(mode="json", exclude_unset=True)["optimizer"] == request["optimizer"]


def test_catalog_availability_is_typed(contract):
    _, raw = example(contract, "listFamilies")
    raw["topologies"] = [
        {
            "topology": "fork_flush",
            "version": 1,
            "label": "experimental",
            "auto": False,
            "availability": "unavailable",
            "cause": "not_served",
        }
    ]
    catalog = FamilyList.model_validate(raw)
    assert catalog.topologies is not None
    assert catalog.topologies[0].availability == "unavailable"
    assert catalog.topologies[0].cause == "not_served"


def test_cli_exposes_optimizer_selector():
    args = parser().parse_args(["run", "task", "--budget", "1.00", "--optimizer", "auto"])
    assert args.optimizer == "auto"


@pytest.mark.parametrize(
    "pin",
    [
        {"topology": "swarms_v2", "version": 2, "params": {"width": 3}},
        {"topology": "", "version": 2},
        {"topology": "bad\ntopology", "version": 2},
        {"topology": "swarms_v2", "version": True},
        {"topology": "swarms_v2", "version": 0},
    ],
)
def test_invalid_pins_and_unknown_parameters_refuse(pin):
    with pytest.raises(ValidationError):
        OptimizerPin.model_validate(pin)


def test_cli_pin_and_sync_convenience_share_generated_schema():
    pin = parse_optimizer("swarms_v2@2")
    with SynthClient("fixture", base_url="https://example.test") as client:
        request = client.swarms.prepare_launch("task", budget="1.00", optimizer=pin)
    assert request.model_dump(mode="json", exclude_unset=True)["optimizer"] == {
        "topology": "swarms_v2",
        "version": 2,
    }


async def test_mcp_optimizer_round_trip_uses_public_sdk(contract):
    sent = []
    _, snapshot = example(contract, "getRun")
    snapshot["optimizer"] = RECEIPT

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot)

    arguments: dict[str, JsonValue] = {
        "body": {
            "message": "task",
            "optimizer": "auto",
            "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
        },
        "idempotency_key": "optimizer-once",
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = json.loads(await call_tool(client, "swarms_run_launch", arguments))
    assert sent[0]["optimizer"] == "auto"
    assert result["optimizer"] == RECEIPT
