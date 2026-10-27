"""MCP must retain caller-selected resources across preflight and launch."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from synth_ai.mcp.catalog import create_run, preflight
from synth_ai.mcp.contracts import RunCreate
from synth_ai.sdk.swarms.types.launch import LaunchRequest


@pytest.mark.parametrize("message_field", ["message", "objective"])
def test_preflight_and_launch_forward_explicit_forge_and_rhodes_grants(message_field):
    body = {
        message_field: "Evaluate the selected workspace",
        "project_id": "execution-project",
        "limits": {
            "maximum_charge": None,
            "wall_time_seconds": 60,
            "maximum_tasks": 1,
            "maximum_parallel_tasks": 1,
        },
        "guidance": [{"id": "goal", "text": "Retain evidence"}],
        "resources": [
            {
                "plugin": "forge.research",
                "roles": ["worker"],
                "limits": {"calls": 5},
                "selection": {"project_ids": ["63000000-0000-4000-8000-000000000001"]},
            },
            {
                "plugin": "rhodes.pool",
                "roles": ["worker"],
                "limits": {"rollouts": 1},
                "selection": {"pool_ids": ["pool-a"]},
            },
        ],
    }
    preflight_args = LaunchRequest.model_validate_json(json.dumps(body))
    create_args = RunCreate.model_validate_json(
        json.dumps({**body, "idempotency_key": "original-key"})
    )
    swarm = SimpleNamespace(
        preflight=AsyncMock(return_value="preview"),
        create=AsyncMock(return_value=SimpleNamespace(run_id="run-1")),
    )
    client = SimpleNamespace(swarms=swarm)

    async def run():
        assert await preflight(client, preflight_args) == "preview"
        assert (await create_run(client, create_args)).run_id == "run-1"

    asyncio.run(run())
    for mock in (swarm.preflight, swarm.create):
        positional, kwargs = mock.await_args
        assert positional == ((body["objective"] if message_field == "objective" else None),)
        assert kwargs["message"] == (body["message"] if message_field == "message" else None)
        assert kwargs["resources"] == preflight_args.resources
        assert kwargs["guidance"] == preflight_args.guidance
        assert kwargs["labels"] == preflight_args.labels
        assert kwargs["repo"] == preflight_args.repo
    assert swarm.create.await_args.kwargs["mutation"].idempotency_key == "original-key"
