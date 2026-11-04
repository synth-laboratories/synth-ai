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


def test_wire_examples(contract):
    for path, item in contract["paths"].items():
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
