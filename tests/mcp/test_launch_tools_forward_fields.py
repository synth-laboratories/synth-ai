"""MCP preflight/create tools forward every LaunchRequest field the SDK accepts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mcp

pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")

from synth_ai.mcp.catalog import create_run, preflight  # noqa: E402
from synth_ai.mcp.contracts import RunCreate  # noqa: E402
from synth_ai.sdk.swarms.types.launch import LaunchRequest  # noqa: E402

LAUNCH = {
    "message": "Run the pool rollout.",
    "project_id": "project_1",
    "limits": {
        "maximum_charge": {"coefficient": 5, "scale": 0, "currency": "USD"},
        "wall_time_seconds": 600,
        "maximum_tasks": 4,
        "maximum_parallel_tasks": 2,
    },
    "guidance": ({"id": "g1", "text": "Prefer the pool worker."},),
    "resources": (
        {
            "plugin": "rhodes.pool",
            "version": 1,
            "roles": ("worker",),
            "limits": {"spend_micros": 1000000, "rollouts": 1},
            "selection": {"pool_ids": ("pool_example",)},
        },
    ),
}


class _Swarms:
    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    async def preflight(self, *args, **kwargs):
        self.calls["preflight"] = {"args": args, **kwargs}
        return "preflight-result"

    async def create(self, *args, **kwargs):
        self.calls["create"] = {"args": args, **kwargs}
        return SimpleNamespace(run_id="run_1")


def _assert_forwarded(call: dict) -> None:
    assert call["message"] == LAUNCH["message"]
    assert [g.id for g in call["guidance"]] == ["g1"]
    (resource,) = call["resources"]
    assert resource.plugin == "rhodes.pool"
    assert resource.roles == ("worker",)
    assert resource.limits == {"spend_micros": 1000000, "rollouts": 1}
    assert resource.selection.pool_ids == ("pool_example",)
    assert call["labels"] is None
    assert call["repo"] is None


def test_preflight_forwards_guidance_and_resources():
    swarms = _Swarms()
    client = SimpleNamespace(swarms=swarms)
    asyncio.run(preflight(client, LaunchRequest.model_validate(LAUNCH)))
    _assert_forwarded(swarms.calls["preflight"])


def test_create_run_forwards_guidance_and_resources():
    swarms = _Swarms()
    client = SimpleNamespace(swarms=swarms)
    args = RunCreate.model_validate({**LAUNCH, "idempotency_key": "idem_1"})
    created = asyncio.run(create_run(client, args))
    assert created.run_id == "run_1"
    _assert_forwarded(swarms.calls["create"])
