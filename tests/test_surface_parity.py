"""Planted defects for the T0 parity detector, independent of today's gaps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "surface_parity", Path(__file__).parents[1] / "scripts/check_surface_parity.py"
)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


def test_public_method_binds_operation():
    assert parity.public_bindings(
        'class Runs:\n async def get(self):\n  return await self._transport.request("getRun")'
    ) == {("getRun", "Runs.get")}


def test_private_method_does_not_prove_customer_reachability():
    assert not parity.public_bindings(
        'class Runs:\n async def _get(self):\n  return await self._transport.request("getRun")'
    )


def test_catalog_string_does_not_prove_sdk_coverage():
    assert not parity.public_bindings('OPERATIONS = {"getRun": "get"}')


def test_private_transport_is_a_failure():
    _, findings = parity.surface_calls(
        'client._transport.invoke("getRun")', surface="cli", path="cli.py"
    )
    assert any(item.code == parity.Code.PRIVATE for item in findings)


def test_raw_http_is_a_failure():
    _, findings = parity.surface_calls("httpx.AsyncClient()", surface="cli", path="cli.py")
    assert any(item.code == parity.Code.TRANSPORT for item in findings)


def test_dictionary_get_does_not_prove_sdk_call():
    methods, _ = parity.surface_calls('payload.get("run")', surface="cli", path="cli.py")
    assert not methods


def test_explicit_receiver_keeps_namespace():
    methods, _ = parity.surface_calls(
        'swarms = client.swarms\nawait swarms.get("run")', surface="cli", path="cli.py"
    )
    assert methods == {"client.swarms.get"}


def fixture(root: Path) -> None:
    (root / "contracts").mkdir()
    (root / "synth_ai/mcp").mkdir(parents=True)
    (root / "contracts/synth-v1.openapi.yaml").write_text(
        "paths:\n  /v1/runs/{run_id}:\n    get:\n      operationId: getRun\n      x-synth-mcp-name: swarms_get\n      x-synth-scope: swarms:read\n"
    )
    (root / "contracts/mcp-tools.json").write_text(
        '{"tools":[{"operationId":"getRun","name":"swarms_get"}]}'
    )
    (root / "synth_ai/aclient.py").write_text(
        "class AsyncSynthClient:\n def __init__(self):\n  self.swarms = AsyncSwarms()\n"
    )
    (root / "synth_ai/swarms.py").write_text(
        'class AsyncSwarms:\n async def get(self):\n  return await self._transport.request("getRun")\n'
    )
    for path in ("synth_ai/cli.py", "synth_ai/mcp/server.py"):
        (root / path).write_text("async def execute(client):\n return await client.swarms.get()\n")


def test_complete_customer_surface_passes(tmp_path: Path):
    fixture(tmp_path)
    assert parity.audit(tmp_path) == ()


def test_planted_missing_cli_operation_fails(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/cli.py").write_text(
        'def execute():\n return {"getRun": "get"}.get("getRun")\n'
    )
    assert any(
        item.surface == "cli" and item.code == parity.Code.MISSING
        for item in parity.audit(tmp_path)
    )


def test_planted_wrong_mcp_name_fails(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "contracts/mcp-tools.json").write_text(
        '{"tools":[{"operationId":"getRun","name":"wrong_name"}]}'
    )
    assert any(item.code == parity.Code.NAME for item in parity.audit(tmp_path))


def test_planted_mcp_name_outside_the_scheme_fails(tmp_path: Path):
    """One scheme: the tool name starts with its scope domain (SYN-5676)."""
    fixture(tmp_path)
    contract = tmp_path / "contracts/synth-v1.openapi.yaml"
    contract.write_text(contract.read_text().replace("swarms_get", "get_run"))
    (tmp_path / "contracts/mcp-tools.json").write_text(
        '{"tools":[{"operationId":"getRun","name":"get_run"}]}'
    )
    assert any(item.code == parity.Code.NAME for item in parity.audit(tmp_path))


def test_planted_private_mcp_dispatch_fails(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text(
        'async def execute(client):\n return await client._transport.invoke("getRun")\n'
    )
    findings = parity.audit(tmp_path)
    assert any(item.surface == "mcp" and item.code == parity.Code.PRIVATE for item in findings)
    assert any(item.surface == "mcp" and item.code == parity.Code.MISSING for item in findings)


def test_imported_mcp_dispatch_is_audited(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text(
        "from .dispatch import dispatch\nasync def call_tool(client):\n return await dispatch(client)\n"
    )
    (tmp_path / "synth_ai/mcp/dispatch.py").write_text(
        "async def dispatch(client):\n return await client.swarms.get()\n"
    )
    assert parity.audit(tmp_path) == ()


def test_unused_mcp_dispatch_does_not_prove_reachability(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text("def execute():\n return 0\n")
    (tmp_path / "synth_ai/mcp/dispatch.py").write_text(
        "async def dispatch(client):\n return await client.swarms.get()\n"
    )
    assert any(
        item.surface == "mcp" and item.code == parity.Code.MISSING
        for item in parity.audit(tmp_path)
    )


def test_private_access_in_imported_dispatch_is_a_failure(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text(
        "from .dispatch import dispatch\nasync def call_tool(client):\n return await dispatch(client)\n"
    )
    (tmp_path / "synth_ai/mcp/dispatch.py").write_text(
        'async def dispatch(client):\n return await client._transport.invoke("getRun")\n'
    )
    assert any(
        item.surface == "mcp" and item.code == parity.Code.PRIVATE
        for item in parity.audit(tmp_path)
    )


def test_imported_but_unused_dispatch_does_not_prove_coverage(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text("from .dispatch import dispatch\n")
    (tmp_path / "synth_ai/mcp/dispatch.py").write_text(
        "async def dispatch(client):\n return await client.swarms.get()\n"
    )
    assert any(
        item.surface == "mcp" and item.code == parity.Code.MISSING
        for item in parity.audit(tmp_path)
    )


def test_dispatch_call_under_wrong_operation_name_fails(tmp_path: Path):
    fixture(tmp_path)
    (tmp_path / "synth_ai/mcp/server.py").write_text(
        'async def execute(client, operation):\n match operation:\n  case "wrongOperation":\n   return await client.swarms.get()\n'
    )
    assert any(
        item.surface == "mcp" and item.operation == "getRun" and item.code == parity.Code.MISSING
        for item in parity.audit(tmp_path)
    )
