from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from synth_ai.core.tunnels.synth_tunnel import RequestContext, SynthTunnelAgent, SynthTunnelLease


class _FakeWs:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.messages.append(payload)


def _sample_lease() -> SynthTunnelLease:
    return SynthTunnelLease(
        lease_id="lease-1",
        route_token="route-1",
        public_base_url="https://st.usesynth.ai",
        public_url="https://st.usesynth.ai/s/route-1",
        agent_url="wss://st.usesynth.ai/ws",
        agent_token="agent-token",
        worker_token="worker-token",
        expires_at=datetime.now(timezone.utc),
        limits={},
        heartbeat={},
    )


def _new_agent() -> SynthTunnelAgent:
    return SynthTunnelAgent(
        lease=_sample_lease(),
        local_host="127.0.0.1",
        local_port=9999,
        stop_event=asyncio.Event(),
    )


def test_try_parse_ws_payload_rejects_invalid_json() -> None:
    assert SynthTunnelAgent._try_parse_ws_payload("{bad json") is None


def test_try_parse_ws_payload_rejects_non_object_json() -> None:
    assert SynthTunnelAgent._try_parse_ws_payload("[1,2,3]") is None


@pytest.mark.asyncio
async def test_handle_req_body_frame_bad_base64_sends_error_and_drops_context() -> None:
    agent = _new_agent()
    ws = _FakeWs()
    rid = "r1"
    agent._contexts[rid] = RequestContext(
        lease_id="lease-1",
        rid=rid,
        method="POST",
        path="/x",
        query="",
        headers=[],
        deadline_ms=1000,
    )

    await agent._handle_req_body_frame(ws, {"rid": rid, "chunk_b64": "AQ"})

    assert rid not in agent._contexts
    assert len(ws.messages) == 1
    payload = json.loads(ws.messages[0])
    assert payload["type"] == "RESP_ERROR"
    assert payload["rid"] == rid
    assert payload["code"] == "BAD_REQUEST_FRAME"


@pytest.mark.asyncio
async def test_handle_req_body_frame_valid_base64_appends_body() -> None:
    agent = _new_agent()
    ws = _FakeWs()
    rid = "r2"
    ctx = RequestContext(
        lease_id="lease-1",
        rid=rid,
        method="POST",
        path="/x",
        query="",
        headers=[],
        deadline_ms=1000,
    )
    agent._contexts[rid] = ctx

    await agent._handle_req_body_frame(ws, {"rid": rid, "chunk_b64": "AQI="})

    assert bytes(ctx.body) == b"\x01\x02"
    assert len(ws.messages) == 0
    assert rid in agent._contexts


@pytest.mark.asyncio
async def test_drain_reconnect_state_clears_contexts_and_cancels_tasks() -> None:
    agent = _new_agent()
    agent._contexts["r1"] = RequestContext(
        lease_id="lease-1",
        rid="r1",
        method="GET",
        path="/x",
        query="",
        headers=[],
        deadline_ms=1000,
    )

    async def _slow() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(_slow())
    agent._tasks["r1"] = task

    await agent._drain_reconnect_state()

    assert agent._contexts == {}
    assert agent._tasks == {}
    assert task.cancelled() or task.done()
