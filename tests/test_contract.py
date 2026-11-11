import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from synth_ai._generated.models import MODELS
from synth_ai._generated.operations import OPERATIONS
from synth_ai._transport import Transport
from synth_ai.mcp.tools import TOOLS

ROOT = Path(__file__).parents[1]


def resolve(doc, value):
    while "$ref" in value:
        reference = value["$ref"]
        value = doc
        for part in reference[2:].split("/"):
            value = value[part]
    return value


def test_generated_drift():
    for script in ["gen_contract.py", "gen_sync.py"]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--check"], check=True)


def test_every_request_has_a_runtime_validator():
    assert {
        op["request_model"] for op in OPERATIONS.values() if op["request_model"]
    } <= MODELS.keys()


@pytest.mark.parametrize(
    "kind,payload",
    [
        (
            "task",
            {
                "name": "sort",
                "description": "Sort integers",
                "input_schema": {"type": "object"},
                "candidate_contract": {},
            },
        ),
        (
            "policy",
            {"name": "sort", "files": {"sort.py": "print(1)"}, "command": ["python", "sort.py"]},
        ),
    ],
)
async def test_union_request_is_validated_and_sent_without_root_wrapper(kind, payload):
    def handle(request):
        assert json.loads(request.content) == payload
        return httpx.Response(
            201, json={"kind": kind, "revision_id": "sha256:" + "a" * 64, "payload": payload}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transport = Transport("https://example.test", "fixture", client=client)
        result = await transport.invoke(
            "registerEvaluationInput", path={"kind": kind}, body=payload, idempotency_key="union"
        )
        assert result.payload.model_dump(mode="json", exclude_unset=True) == payload


def test_wire_examples(contract):
    for _path, item in contract["paths"].items():
        for operation in item.values():
            for response in operation.get("responses", {}).values():
                response = resolve(contract, response)
                for media in response.get("content", {}).values():
                    schema = media.get("schema", {})
                    example = media.get("example")
                    name = schema.get("$ref", "").split("/")[-1]
                    if name in MODELS and example is not None:
                        model = MODELS[name].model_validate(example)
                        assert json.loads(model.model_dump_json(exclude_unset=True)) == example


def test_mcp_inventory(contract):
    declared = {
        op["operationId"]: op["x-synth-mcp-name"]
        for path, item in contract["paths"].items()
        for op in item.values()
        if isinstance(op, dict) and "x-synth-mcp-name" in op
    }
    assert {tool["operationId"]: tool["name"] for tool in TOOLS} == declared
    assert TOOLS == json.loads((ROOT / "contracts/mcp-tools.json").read_text())["tools"]


@pytest.mark.parametrize("oid", OPERATIONS)
async def test_every_operation_on_wire(contract, oid):
    op = OPERATIONS[oid]
    source = contract["paths"][op["path"]][op["method"].lower()]
    code, response = next(
        (int(code), resolve(contract, response))
        for code, response in source["responses"].items()
        if code.startswith("2")
    )
    content = response.get("content", {})
    expected = content.get("application/json", {}).get("example")

    def handle(request):
        assert request.method == op["method"]
        if op["auth"]:
            assert request.headers["Authorization"] == "Bearer fixture"
        if op["idempotent"]:
            assert request.headers["Idempotency-Key"] == "stable-key"
        if expected is not None:
            return httpx.Response(code, json=expected)
        return httpx.Response(code, content=b"output" if op["binary"] else b"")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transport = Transport("https://example.test", "fixture", client=client)
        result = await transport.invoke(
            oid, path={name: "fixture" for name in op["path_params"]}, idempotency_key="stable-key"
        )
        if expected is not None:
            assert isinstance(result, BaseModel)
            assert result.model_dump(mode="json", exclude_unset=True) == expected
        elif op["binary"]:
            assert result == b"output"
        else:
            assert result is None


def test_generated_mcp_name_scope_and_argument_parity():
    hosted = json.loads((ROOT / "contracts/mcp-tools.json").read_text())
    by_name = {tool["name"]: tool for tool in TOOLS}
    assert len(by_name) == len(hosted["tools"]) == 30
    assert {
        "swarms_run_pause",
        "swarms_run_resume",
        "swarms_run_tasks",
        "usage_for_run",
        "swarms_run_wait",
        "swarms_run_result",
    } <= set(by_name)
    for tool in hosted["tools"]:
        local = by_name.pop(tool["name"])
        assert local["inputSchema"] == tool["inputSchema"]
        assert OPERATIONS[local["operationId"]]["scope"] == tool["scope"]
    assert not by_name


def test_a_nullable_constrained_field_accepts_null():
    """A pattern must apply only to the string arm; pydantic raises on null when it reaches the validator."""
    bucket = {
        "charge": {"amount": "0", "currency": "USD"},
        "input_tokens": 0,
        "output_tokens": 0,
        "records": 0,
    }
    assert MODELS["UsageKeyBucket"].model_validate({**bucket, "key_id": None}).key_id is None


def test_generated_models_never_constrain_null():
    assert "Annotated[None" not in (ROOT / "synth_ai/_generated/models.py").read_text()


def test_nullable_subscription_response_round_trips_null():
    response = MODELS[OPERATIONS["getSubscription"]["response_model"]].model_validate(None)
    assert response.model_dump(mode="json") is None
