"""SYN-6058: Local MCP Events (SEP-3415) and `synth watch <run_id>` CLI tests."""

from __future__ import annotations

import json
from typing import cast

import anyio
import httpx
import pytest

from synth_ai import AsyncSynthClient, cli
from synth_ai.mcp import events

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"


def _run_snapshot_example(contract: dict[str, object], *, state: str) -> dict[str, object]:
    paths = contract["paths"]
    assert isinstance(paths, dict)
    snap = json.loads(
        json.dumps(
            paths["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"]["application/json"][
                "example"
            ]
        )
    )
    assert isinstance(snap, dict)
    closed = state in {"completed", "partial", "failed", "cancelled"}
    snap.update(
        run_id=RUN_ID,
        state=state,
        created_at="2026-11-27T15:00:00Z",
        started_at="2026-11-27T15:00:01Z",
        closed_at="2026-11-27T15:00:10Z" if closed else None,
        execution_closed=closed,
        accounting_final=closed,
        cleanup_complete=state == "completed",
        result_available=state == "completed",
    )
    return cast("dict[str, object]", snap)


def _sample_events() -> list[dict[str, object]]:
    return [
        {
            "event_id": "evt_0000000000000000000001",
            "run_id": RUN_ID,
            "sequence": 1,
            "kind": "objective.progress",
            "original_kind": "objective.progress",
            "usage": None,
            "observed_at": "2026-11-27T15:00:01Z",
            "approval_id": None,
            "payload": {
                "objective_id": "goal",
                "name": "accuracy",
                "current": 0.65,
                "best": 0.65,
                "target": 1.0,
                "direction": "maximize",
                "met": False,
                "percent": 65,
                "samples": 2,
                "rollout_id": "r1",
                "receipt_digest": "a" * 64,
            },
        },
        {
            "event_id": "evt_0000000000000000000002",
            "run_id": RUN_ID,
            "sequence": 2,
            "kind": "limit.warning",
            "original_kind": "limit.warning",
            "usage": None,
            "observed_at": "2026-11-27T15:00:02Z",
            "approval_id": None,
            "payload": {
                "revision": 1,
                "at_ms": 1795800002000,
                "scope": "run",
                "dimension": "budget_usd",
                "cap": 1000,
                "used": 800,
                "reserved": 50,
                "unknown_reserved": 0,
            },
        },
        {
            "event_id": "evt_0000000000000000000003",
            "run_id": RUN_ID,
            "sequence": 3,
            "kind": "run.message.posted",
            "original_kind": "run.message.posted",
            "usage": None,
            "observed_at": "2026-11-27T15:00:03Z",
            "approval_id": None,
            "payload": {
                "message_id": "msg_AAAAAAAAAAAAAAAAAAAAAA",
                "state": {"kind": "posted"},
                "author": "manager",
            },
        },
        {
            "event_id": "evt_0000000000000000000004",
            "run_id": RUN_ID,
            "sequence": 4,
            "kind": "approval.requested",
            "original_kind": "approval.requested",
            "usage": None,
            "observed_at": "2026-11-27T15:00:04Z",
            "approval_id": "apr-1",
            "payload": {
                "approval_id": "apr-1",
                "action_kind": "git_push",
                "summary": "Push feature branch",
                "requested_by": {"actor_id": "planner", "actor_kind": "actor"},
                "requested_at": "2026-11-27T15:00:04Z",
                "timeout_ms": 30000,
                "expires_at": "2026-11-27T15:00:34Z",
                "revision": 1,
                "status": "pending",
            },
        },
        {
            "event_id": "evt_0000000000000000000005",
            "run_id": RUN_ID,
            "sequence": 5,
            "kind": "approval.decided",
            "original_kind": "approval.decided",
            "usage": None,
            "observed_at": "2026-11-27T15:00:05Z",
            "approval_id": "apr-1",
            "payload": {
                "approval_id": "apr-1",
                "decision": "approve",
                "status": "approved",
                "operation_id": "op-1",
                "decided_by": {"actor_id": "user:u1", "actor_kind": "human"},
                "decided_at": "2026-11-27T15:00:05Z",
                "revision": 2,
                "reason": "ship it",
            },
        },
        {
            "event_id": "evt_0000000000000000000006",
            "run_id": RUN_ID,
            "sequence": 6,
            "kind": "research.artifact_published",
            "original_kind": "research.artifact_published",
            "usage": None,
            "observed_at": "2026-11-27T15:00:06Z",
            "approval_id": None,
            "payload": {
                "artifact_id": "art_1",
                "notebook_id": "nb_1",
                "title": "Loss Curve",
                "kind": "chart",
                "origin": "agent",
                "revision": 1,
                "revision_id": "rev_1",
                "digest_sha256": "b" * 64,
                "author_kind": "swarm_actor",
                "summary": "Initial chart",
            },
        },
        {
            "event_id": "evt_0000000000000000000007",
            "run_id": RUN_ID,
            "sequence": 7,
            "kind": "research.experiment_created",
            "original_kind": "research.experiment_created",
            "usage": None,
            "observed_at": "2026-11-27T15:00:07Z",
            "approval_id": None,
            "payload": {
                "experiment_id": "exp_1",
                "notebook_id": "nb_1",
                "title": "LR Sweep",
                "revision": 1,
                "arms": ["baseline", "low-lr"],
                "hypothesis": "lower lr helps",
            },
        },
        {
            "event_id": "evt_0000000000000000000008",
            "run_id": RUN_ID,
            "sequence": 8,
            "kind": "research.trial_logged",
            "original_kind": "research.trial_logged",
            "usage": None,
            "observed_at": "2026-11-27T15:00:08Z",
            "approval_id": None,
            "payload": {
                "experiment_id": "exp_1",
                "trial_id": "tr_1",
                "arm": "low-lr",
                "state": "succeeded",
                "depth": 1,
            },
        },
        {
            "event_id": "evt_0000000000000000000009",
            "run_id": RUN_ID,
            "sequence": 9,
            "kind": "research.hypothesis_created",
            "original_kind": "research.hypothesis_created",
            "usage": None,
            "observed_at": "2026-11-27T15:00:09Z",
            "approval_id": None,
            "payload": {
                "entry_id": "log_1",
                "notebook_id": "nb_1",
                "category": "hypothesis",
                "author_kind": "swarm_actor",
                "author_label": "planner",
                "sync_state": "synced",
                "correction_of": None,
                "artifact_ids": ["art_1"],
            },
        },
    ]


async def _raw_rpc_collect(server, requests):
    """Send JSON-RPC requests and collect all responses and notifications until all IDs reply."""
    from mcp.shared.memory import create_client_server_memory_streams
    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    messages = []
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                )
            )
            try:
                async with client_read, client_write:
                    for req in requests:
                        req_id = req["id"]
                        await client_write.send(
                            SessionMessage(
                                message=JSONRPCMessage(JSONRPCRequest.model_validate(req))
                            )
                        )
                        while True:
                            reply = await client_read.receive()
                            assert isinstance(reply, SessionMessage)
                            dumped = reply.message.root.model_dump(mode="json", by_alias=True)
                            messages.append(dumped)
                            if dumped.get("id") == req_id:
                                break
            finally:
                tg.cancel_scope.cancel()
    return messages


async def test_mcp_events_capabilities_list_poll_and_refusals(
    contract: dict[str, object],
) -> None:
    pytest.importorskip("mcp")
    from synth_ai.mcp.server import build_server

    sample = _sample_events()
    page0 = {
        "items": [sample[0], sample[0], sample[3], sample[4]],
        "has_more": True,
        "next_cursor": "up-1",
        "resume_cursor": "up-1",
    }
    page1 = {
        "items": [sample[5], sample[6]],
        "has_more": False,
        "resume_cursor": "up-2",
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/events":
            cursor = request.url.params.get("cursor")
            if cursor == "up-expired":
                return httpx.Response(410, json={"code": "cursor_expired", "detail": "expired"})
            if cursor is None:
                return httpx.Response(200, json=page0)
            if cursor == "up-1":
                return httpx.Response(200, json=page1)
            if cursor == "up-2":
                return httpx.Response(
                    200, json={"items": [], "has_more": False, "resume_cursor": "up-2"}
                )
            return httpx.Response(422, json={"code": "invalid_cursor", "detail": "bad cursor"})
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "run_not_found", "detail": "not found"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        replies = await _raw_rpc_collect(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
                {"jsonrpc": "2.0", "id": "list", "method": "events/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": "poll-approvals-1",
                    "method": "events/poll",
                    "params": {
                        "event": "swarms.run.approval_requested",
                        "input": {"run_id": RUN_ID, "include_decisions": True},
                        "limit": 1,
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "poll-milestones",
                    "method": "events/poll",
                    "params": {
                        "event": "swarms.run.milestone",
                        "input": {"run_id": RUN_ID},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "poll-bad-event",
                    "method": "events/poll",
                    "params": {
                        "event": "swarms.run.bogus",
                        "input": {"run_id": RUN_ID},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "poll-bad-cursor",
                    "method": "events/poll",
                    "params": {
                        "event": "swarms.run.progress",
                        "input": {"run_id": RUN_ID},
                        "cursor": "bogus-cursor",
                    },
                },
            ],
        )

        by_id = {msg["id"]: msg for msg in replies if "id" in msg}
        caps = by_id["init"]["result"]["capabilities"]
        assert caps["extensions"]["io.modelcontextprotocol/skills"] == {"directoryRead": True}
        assert caps["extensions"]["io.modelcontextprotocol/events"] == {"poll": True, "push": True}
        assert "webhook" not in caps["extensions"]["io.modelcontextprotocol/events"]

        listed = by_id["list"]["result"]["events"]
        assert [item["name"] for item in listed] == [
            "swarms.run.milestone",
            "swarms.run.progress",
            "swarms.run.approval_requested",
            "swarms.run.message_posted",
            "swarms.run.limit_alert",
            "swarms.research.updated",
        ]
        for item in listed:
            assert item["deliveryModes"] == ["poll", "push"]
            assert item["inputSchema"]["additionalProperties"] is False
            assert "run_id" in item["inputSchema"]["required"]
            assert item["payloadSchema"]["type"] == "object"

        # Approval poll split across limit=1 then resumed via cursor.
        first_poll = by_id["poll-approvals-1"]["result"]
        assert len(first_poll["events"]) == 1
        assert first_poll["events"][0]["payload"]["kind"] == "approval.requested"
        assert first_poll["hasMore"] is True
        assert first_poll["truncated"] is False

        second_poll = await events.poll_events(
            client,
            {
                "event": "swarms.run.approval_requested",
                "input": {"run_id": RUN_ID, "include_decisions": True},
                "cursor": first_poll["cursor"],
                "limit": 1,
            },
        )
        second_events = second_poll["events"]
        assert isinstance(second_events, list)
        assert len(second_events) == 1
        second_ev = second_events[0]
        assert isinstance(second_ev, dict)
        second_payload = second_ev["payload"]
        assert isinstance(second_payload, dict)
        assert second_payload["kind"] == "approval.decided"

        # Milestones poll returns all 3 reached milestones for completed run.
        ms_result = by_id["poll-milestones"]["result"]
        assert len(ms_result["events"]) == 3
        assert [ev["payload"]["milestone"] for ev in ms_result["events"]] == [
            "execution_closed",
            "accounting_final",
            "cleanup_complete",
        ]
        ms_replay = await events.poll_events(
            client,
            {
                "event": "swarms.run.milestone",
                "input": {"run_id": RUN_ID},
                "cursor": ms_result["cursor"],
            },
        )
        assert ms_replay["events"] == []

        # Truncation recovery on HTTP 410 Gone.
        expired_cursor = events.encode_cursor(
            events.EventCursor(
                event="swarms.run.progress",
                run_id=RUN_ID,
                upstream="up-expired",
            )
        )
        trunc = await events.poll_events(
            client,
            {
                "event": "swarms.run.progress",
                "input": {"run_id": RUN_ID},
                "cursor": expired_cursor,
            },
        )
        assert trunc["truncated"] is True
        trunc_events = trunc["events"]
        assert isinstance(trunc_events, list)
        assert len(trunc_events) == 1  # deduped sample[0]

        # Unknown event & malformed cursor refuse fail-closed with -32602.
        assert by_id["poll-bad-event"]["error"]["code"] == -32602
        assert by_id["poll-bad-cursor"]["error"]["code"] == -32602


async def test_mcp_events_stream_pushes_notifications_over_stdio(
    contract: dict[str, object],
) -> None:
    pytest.importorskip("mcp")
    from synth_ai.mcp.server import build_server

    sample = _sample_events()
    seen_last_event_id: list[str | None] = []
    sse_body = (
        "id: 1\nevent: event\ndata: "
        + json.dumps(sample[0])
        + "\n\n"
        + "id: 2\nevent: event\ndata: "
        + json.dumps(sample[5])
        + "\n\n"
        + "id: 2.3\nevent: milestone\ndata: "
        + json.dumps(
            {
                "milestone": "cleanup_complete",
                "run_id": RUN_ID,
                "state": "completed",
                "execution_closed": True,
                "accounting_final": True,
                "cleanup_complete": True,
                "result_available": True,
                "failure": None,
                "final": True,
            }
        )
        + "\n\n"
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/events":
            seen_last_event_id.append(request.headers.get("Last-Event-ID"))
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "run_not_found", "detail": "not found"})

    resume_cursor = events.encode_cursor(
        events.EventCursor(event="", run_id=RUN_ID, stream_resume_id="0")
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        messages = await _raw_rpc_collect(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "stream-1",
                    "method": "events/stream",
                    "params": {
                        "events": [
                            "swarms.run.progress",
                            "swarms.research.updated",
                            "swarms.run.milestone",
                        ],
                        "input": {"run_id": RUN_ID},
                        "cursor": resume_cursor,
                        "timeoutSeconds": 5,
                    },
                },
            ],
        )

    assert seen_last_event_id == ["0"]
    notifications = [m for m in messages if m.get("method") == "notifications/events/event"]
    assert [n["params"]["event"] for n in notifications] == [
        "swarms.run.progress",
        "swarms.research.updated",
        "swarms.run.milestone",
    ]
    final_reply = next(m for m in messages if m.get("id") == "stream-1")
    assert final_reply["result"]["status"] == "completed"
    assert final_reply["result"]["delivered"] == 3
    assert final_reply["result"]["run_id"] == RUN_ID


def test_cli_watch_human_and_json_streaming_and_once(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    sample = _sample_events()
    milestone_frame = {
        "milestone": "cleanup_complete",
        "run_id": RUN_ID,
        "state": "completed",
        "execution_closed": True,
        "accounting_final": True,
        "cleanup_complete": True,
        "result_available": True,
        "failure": None,
        "final": True,
    }
    sse_frames = (
        "".join(
            f"id: {idx + 1}\nevent: event\ndata: {json.dumps(ev)}\n\n"
            for idx, ev in enumerate(sample)
        )
        + f"id: {len(sample)}.3\nevent: milestone\ndata: {json.dumps(milestone_frame)}\n\n"
    )

    seen_headers: list[str | None] = []
    seen_cursors: list[str | None] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/events":
            if request.headers.get("Accept") == "text/event-stream":
                seen_headers.append(request.headers.get("Last-Event-ID"))
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=sse_frames.encode("utf-8"),
                )
            seen_cursors.append(request.url.params.get("cursor"))
            return httpx.Response(
                200,
                json={"items": sample, "has_more": False, "resume_cursor": "r-end"},
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "run_not_found", "detail": "not found"})

    # 1. Human streaming output with --after
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code = cli.main(
        ["--base-url", "http://127.0.0.1:9", "watch", RUN_ID, "--after", "0"],
        http_client=http,
    )
    assert code == 0
    assert seen_headers == ["0"]
    out = capsys.readouterr().out
    assert "[progress] accuracy 65%" in out
    assert "[limit:warning] budget_usd" in out
    assert "[message] run.message.posted" in out
    assert "[approval:requested] apr-1" in out
    assert "[approval:decided] apr-1 decision=approve" in out
    assert '[research:artifact] art_1 "Loss Curve"' in out
    assert '[research:experiment] exp_1 "LR Sweep"' in out
    assert "[research:trial] exp_1/tr_1" in out
    assert "[research:hypothesis] log_1" in out
    assert "[milestone] cleanup_complete state=completed" in out

    # 2. --json (NDJSON) streaming mode
    http_json = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code_json = cli.main(
        ["--base-url", "http://127.0.0.1:9", "watch", RUN_ID, "--json"],
        http_client=http_json,
    )
    assert code_json == 0
    ndjson_lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(ndjson_lines) == len(sample) + 1
    assert ndjson_lines[0]["kind"] == "objective.progress"
    assert ndjson_lines[-1]["milestone"] == "cleanup_complete"
    assert ndjson_lines[-1]["final"] is True

    # 3. --once --after --json deterministic snapshot mode
    http_once = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code_once = cli.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            "watch",
            RUN_ID,
            "--once",
            "--after",
            "cur-1",
            "--json",
        ],
        http_client=http_once,
    )
    assert code_once == 0
    assert seen_cursors == ["cur-1"]
    once_lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    # 9 events + 3 reached milestones (execution_closed, accounting_final, cleanup_complete)
    assert len(once_lines) == len(sample) + 3
    assert [m["milestone"] for m in once_lines[-3:]] == [
        "execution_closed",
        "accounting_final",
        "cleanup_complete",
    ]
