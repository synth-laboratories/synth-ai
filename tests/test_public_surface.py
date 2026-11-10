import importlib.util
import subprocess
import sys

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai._generated.operations import OPERATIONS
from synth_ai.cli import COMMANDS
from synth_ai.mcp.tools import TOOLS


async def test_swarms_only_public_surface(contract):
    assert (
        sum(
            "operationId" in op
            for path in contract["paths"].values()
            for op in path.values()
            if isinstance(op, dict)
        )
        == 59
    )
    assert len(OPERATIONS) == 46
    assert len(TOOLS) == 23
    assert "inference" not in COMMANDS
    assert importlib.util.find_spec("synth_ai.inference") is None
    assert all("inference" not in op["path"] for op in OPERATIONS.values())
    assert all("inference" not in tool["name"] for tool in TOOLS)
    client = AsyncSynthClient("fixture", base_url="https://example.test")
    assert not hasattr(client, "inference")
    with SynthClient("fixture", base_url="https://example.test") as client:
        assert not hasattr(client, "inference")
    result = subprocess.run(
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
