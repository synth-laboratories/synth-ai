"""A run waiting for capacity reads its queue standing (contract 1.32.0, SYN-5720).

Core adds RunSnapshot.queue on launch and single-run reads of a queued run and
omits it once the run is admitted. The SDK model, the CLI run read and the MCP
swarms_run_get tool all carry it.
"""

import json
from pathlib import Path

import httpx

from synth_ai import AsyncSynthClient
from synth_ai._cli_view import snapshot
from synth_ai._generated.models import RunSnapshot
from synth_ai.mcp.server import call_tool

FIXTURES = Path(__file__).parent / "fixtures"
RUNNING = json.loads((FIXTURES / "core_limit_paused_run_snapshot.json").read_text())
STANDING = {
    "position": 3,
    "ahead": 2,
    "estimated_start_at": None,
    "expires_at": "2026-11-17T05:00:00Z",
}
QUEUED = {**RUNNING, "state": "queued", "queue": STANDING}


def _client(http: httpx.AsyncClient) -> AsyncSynthClient:
    return AsyncSynthClient("fixture", base_url="https://example.test", http_client=http)


async def test_get_and_mcp_read_carry_queue_standing():
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=QUEUED)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with _client(http) as client:
            run = await client.swarms.get(QUEUED["run_id"])
            served = json.loads(
                await call_tool(client, "swarms_run_get", {"run_id": QUEUED["run_id"]})
            )
    assert run.queue is not None
    assert (run.queue.position, run.queue.ahead) == (3, 2)
    assert run.queue.estimated_start_at is None
    assert served["queue"] == STANDING


def test_cli_run_read_shows_standing_only_while_queued():
    queued = snapshot(RunSnapshot.model_validate(QUEUED))
    assert "position 3 (2 ahead)" in queued
    assert "2026-11-17T05:00:00Z" in queued
    running = snapshot(RunSnapshot.model_validate(RUNNING))
    assert "queue" not in running and "expires" not in running
