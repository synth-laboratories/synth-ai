import importlib.util
import subprocess
import sys

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai._generated.operations import OPERATIONS
from synth_ai.cli import COMMANDS
from synth_ai.mcp.tools import TOOLS


async def test_swarms_only_public_surface(contract):
    # Counts come from the pinned contract, never a hardcoded number: every
    # implemented /v1 operation, and each operation outside /v1 the contract publishes as
    # an MCP tool (revokeOAuthGrant, 1.40.0), is one SDK operation and exactly one MCP tool.
    v1_operations = [
        op["operationId"]
        for path, item in contract["paths"].items()
        for op in item.values()
        if isinstance(op, dict)
        and op.get("x-synth-status") == "implemented"
        and (path.startswith("/v1/") or "x-synth-mcp-name" in op)
    ]
    assert len(v1_operations) == len(set(v1_operations))
    assert set(OPERATIONS) == set(v1_operations)
    assert len(TOOLS) == len(OPERATIONS)
    assert sorted(tool["operationId"] for tool in TOOLS) == sorted(OPERATIONS)
    assert "inference" not in COMMANDS
    assert importlib.util.find_spec("synth_ai.inference") is None
    assert all("inference" not in op["path"] for op in OPERATIONS.values())
    assert all("inference" not in tool["name"] for tool in TOOLS)
    client = AsyncSynthClient("fixture", base_url="https://example.test")
    assert not hasattr(client, "inference")
    with SynthClient("fixture", base_url="https://example.test") as client:
        assert not hasattr(client, "inference")
    result = subprocess.run(  # noqa: ASYNC221 -- sync CLI invocation in unit test
        [sys.executable, "-m", "synth_ai.cli", "inference", "session"], capture_output=True
    )
    assert result.returncode == 1
    assert b"invalid choice" in result.stderr


def test_only_implemented_operations_are_callable(contract):
    planned = {
        op["operationId"]
        for item in contract["paths"].values()
        for op in item.values()
        if isinstance(op, dict) and op.get("x-synth-status") == "planned"
    }
    assert not planned.intersection(OPERATIONS)
    assert {
        "pauseRun",
        "resumeRun",
        "getRunProgram",
        "getRunConfiguration",
        "listRunTasks",
        "createUsageExport",
        "getUsageExport",
    } <= OPERATIONS.keys()


def test_sdk_inventory_covers_every_implemented_v1_operation(contract):
    expected = {
        op["operationId"]
        for path, item in contract["paths"].items()
        for op in item.values()
        if isinstance(op, dict)
        and op.get("x-synth-status") == "implemented"
        and (path.startswith("/v1/") or "x-synth-mcp-name" in op)
    }
    assert set(OPERATIONS) == expected


def test_evaluation_surface_is_removed():
    assert importlib.util.find_spec("synth_ai.evaluations") is None
    assert "evaluations" not in COMMANDS
    assert all("/evaluations" not in op["path"] for op in OPERATIONS.values())
    assert all(not tool["name"].startswith("swarms_evaluation_") for tool in TOOLS)
    with SynthClient("fixture", base_url="https://example.test") as client:
        assert not hasattr(client, "evaluations")
    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "evaluations", "list"], capture_output=True
    )
    assert result.returncode == 1
    assert b"invalid choice" in result.stderr
