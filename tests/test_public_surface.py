import importlib.util
import inspect
import re
import subprocess
import sys

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai._generated.operations import OPERATIONS
from synth_ai._sync import AccountClient, SwarmsClient
from synth_ai.account import AsyncAccount
from synth_ai.cli import COMMANDS
from synth_ai.mcp.tools import TOOLS
from synth_ai.swarms import AsyncSwarms


async def test_swarms_only_public_surface():
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


def test_sdk_inventory_covers_every_implemented_v1_operation(contract):
    expected = {
        op["operationId"]
        for path, item in contract["paths"].items()
        if path.startswith("/v1/")
        for op in item.values()
        if isinstance(op, dict) and op.get("x-synth-status") == "implemented"
    }
    assert set(OPERATIONS) == expected


def test_evaluation_surface_is_removed():
    assert importlib.util.find_spec("synth_ai.evaluations") is None
    assert "evaluations" not in COMMANDS
    retired_paths = re.compile(r"^/v1/(evaluations?|evaluation-inputs|index)(?:/|$)")
    retired_tools = re.compile(r"(^|_)(evaluations?|index)(_|$)")
    assert all(not retired_paths.search(op["path"]) for op in OPERATIONS.values())
    assert all(not retired_tools.search(tool["name"]) for tool in TOOLS)
    with SynthClient("fixture", base_url="https://example.test") as client:
        assert not hasattr(client, "evaluations")
    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "evaluations", "list"], capture_output=True
    )
    assert result.returncode == 1
    assert b"invalid choice" in result.stderr


def test_sync_public_methods_retain_async_documentation():
    assert inspect.getdoc(AsyncSwarms.launch)
    assert inspect.getdoc(AsyncAccount.get)
    for sync, async_ in ((SwarmsClient, AsyncSwarms), (AccountClient, AsyncAccount)):
        for name, method in vars(sync).items():
            if not name.startswith("_") and inspect.isfunction(method):
                assert inspect.getdoc(method) == inspect.getdoc(getattr(async_, name))
