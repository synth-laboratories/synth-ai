"""Tests for SYN-6057: typed approval.*/research.* RunEvents and stdio MCP
progress notifications, MCP Tasks (2026-07-28), and approval elicitation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai.models import (
    ResearchArtifactEventPayload,
    ResearchExperimentEventPayload,
    ResearchLogEventPayload,
    ResearchTrialEventPayload,
    RunApprovalDecidedPayload,
    RunApprovalRequestedPayload,
    RunEvent,
)

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams, ElicitResult, LoggingMessageNotificationParams

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"
EVT_ID = "evt_AAAAAAAAAAAAAAAAAAAAAA"


def test_typed_approval_and_research_run_event_models() -> None:
    requested = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 4,
            "kind": "approval.requested",
            "original_kind": "approval.parked",
            "usage": None,
            "observed_at": "2026-11-27T15:00:00Z",
            "approval_id": "apr-1",
            "payload": {
                "approval_id": "apr-1",
                "action_kind": "git_push",
                "summary": "Push branch",
                "requested_by": {"actor_id": "planner", "actor_kind": "actor"},
                "requested_at": "2026-11-27T15:00:00Z",
                "timeout_ms": 30000,
                "expires_at": "2026-11-27T15:00:30Z",
                "revision": 1,
                "status": "pending",
            },
        }
    )
    assert isinstance(requested.payload, RunApprovalRequestedPayload)
    assert requested.payload.approval_id == "apr-1"
    assert requested.approval_id == "apr-1"

    decided = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
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
        }
    )
    assert isinstance(decided.payload, RunApprovalDecidedPayload)
    assert decided.payload.decision == "approve"

    artifact = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 5,
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
                "digest_sha256": "a" * 64,
                "author_kind": "swarm_actor",
                "summary": "Initial chart",
            },
        }
    )
    assert isinstance(artifact.payload, ResearchArtifactEventPayload)

    experiment = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 5,
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
        }
    )
    assert isinstance(experiment.payload, ResearchExperimentEventPayload)

    trial = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 5,
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
        }
    )
    assert isinstance(trial.payload, ResearchTrialEventPayload)

    hypothesis = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 5,
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
        }
    )
    assert isinstance(hypothesis.payload, ResearchLogEventPayload)

    correction = RunEvent.model_validate(
        {
            "event_id": EVT_ID,
            "run_id": RUN_ID,
            "sequence": 5,
            "kind": "research.correction_recorded",
            "original_kind": "research.correction_recorded",
            "usage": None,
            "observed_at": "2026-11-27T15:00:10Z",
            "approval_id": None,
            "payload": {
                "entry_id": "log_2",
                "notebook_id": "nb_1",
                "category": "correction",
                "author_kind": "human",
                "author_label": "usr_1",
                "sync_state": "synced",
                "correction_of": "log_1",
                "artifact_ids": [],
            },
        }
    )
    assert isinstance(correction.payload, ResearchLogEventPayload)
    assert correction.payload.correction_of == "log_1"


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
    snap.update(
        run_id=RUN_ID,
        state=state,
        created_at="2026-11-27T15:00:00Z",
        started_at="2026-11-27T15:00:01Z",
        closed_at="2026-11-27T15:00:10Z" if state in {"completed", "failed", "cancelled"} else None,
        execution_closed=state in {"completed", "failed", "cancelled"},
        accounting_final=state in {"completed", "failed", "cancelled"},
        cleanup_complete=state in {"completed", "failed", "cancelled"},
    )
    return cast("dict[str, object]", snap)


async def test_stdio_mcp_tasks_launch_get_result_cancel(contract: dict[str, object]) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import (
        CallToolRequest,
        CallToolRequestParams,
        CancelTaskRequest,
        CancelTaskRequestParams,
        CancelTaskResult,
        ClientRequest,
        CreateTaskResult,
        GetTaskPayloadRequest,
        GetTaskPayloadRequestParams,
        GetTaskPayloadResult,
        GetTaskRequest,
        GetTaskRequestParams,
        GetTaskResult,
        TaskMetadata,
    )

    from synth_ai.mcp.server import build_server

    state_box = {"state": "running", "pending_approvals": 0}
    paths = contract["paths"]
    assert isinstance(paths, dict)
    result_example = json.loads(
        json.dumps(
            paths["/v1/runs/{run_id}/result"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        )
    )
    result_example["run_id"] = RUN_ID

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/runs":
            return httpx.Response(202, json=_run_snapshot_example(contract, state="queued"))
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(
                200, json=_run_snapshot_example(contract, state=str(state_box["state"]))
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/approvals":
            items = (
                [
                    {
                        "approval_id": "apr-1",
                        "run_id": RUN_ID,
                        "action_kind": "git_push",
                        "summary": "Push branch",
                        "requested_by": {"actor_id": "planner", "actor_kind": "actor"},
                        "requested_at": "2026-11-27T15:00:00Z",
                        "timeout_ms": 30000,
                        "expires_at": "2026-11-27T15:00:30Z",
                        "revision": 1,
                        "status": "pending",
                    }
                ]
                if state_box["pending_approvals"]
                else []
            )
            return httpx.Response(200, json={"items": items, "has_more": False})
        if request.method == "POST" and request.url.path == f"/v1/runs/{RUN_ID}/cancel":
            state_box["state"] = "cancelled"
            return httpx.Response(202, json=_run_snapshot_example(contract, state="cancelled"))
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/result":
            return httpx.Response(200, json=result_example)
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            tools = (await session.list_tools()).tools
            launch_tool = next(t for t in tools if t.name == "swarms_run_launch")
            assert launch_tool.execution is not None
            assert launch_tool.execution.taskSupport == "optional"

            # 1. Task-augmented swarms_run_launch -> CreateTaskResult
            created = await session.send_request(
                ClientRequest(
                    CallToolRequest(
                        params=CallToolRequestParams(
                            name="swarms_run_launch",
                            arguments={
                                "idempotency_key": "mcp-task-1",
                                "body": {
                                    "message": "run task",
                                    "limits": {
                                        "maximum_charge": {"amount": "0.10", "currency": "USD"},
                                        "wall_time_seconds": 30,
                                    },
                                },
                            },
                            task=TaskMetadata(ttl=60000),
                        )
                    )
                ),
                CreateTaskResult,
            )
            assert created.task.taskId == RUN_ID
            assert created.task.status == "working"

            # 2. tasks/get with pending approval -> input_required
            state_box["pending_approvals"] = 1
            status_res = await session.send_request(
                ClientRequest(GetTaskRequest(params=GetTaskRequestParams(taskId=RUN_ID))),
                GetTaskResult,
            )
            assert status_res.status == "input_required"

            # 3. tasks/cancel -> cancelled
            cancelled = await session.send_request(
                ClientRequest(CancelTaskRequest(params=CancelTaskRequestParams(taskId=RUN_ID))),
                CancelTaskResult,
            )
            assert cancelled.status == "cancelled"

            # 4. tasks/result on completed run -> GetTaskPayloadResult with related-task meta
            state_box["state"] = "completed"
            state_box["pending_approvals"] = 0
            payload = await session.send_request(
                ClientRequest(
                    GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(taskId=RUN_ID))
                ),
                GetTaskPayloadResult,
            )
            dumped = payload.model_dump(mode="json", by_alias=True)
            assert dumped["_meta"]["io.modelcontextprotocol/related-task"]["taskId"] == RUN_ID
            assert dumped["isError"] is False


async def test_stdio_mcp_wait_progress_logging_and_elicitation(
    contract: dict[str, object],
) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import ElicitResult, TextContent

    from synth_ai.mcp.server import build_server

    decided_requests: list[dict[str, object]] = []
    progress_updates: list[tuple[float, float | None, str | None]] = []
    log_messages: list[str] = []
    elicited_prompts: list[str] = []

    sse_body = (
        "id: 1\n"
        "event: event\n"
        + "data: "
        + json.dumps(
            {
                "event_id": EVT_ID,
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
                    "current": 0.75,
                    "best": 0.75,
                    "target": 1.0,
                    "direction": "maximize",
                    "met": False,
                    "percent": 75,
                    "samples": 2,
                    "rollout_id": "r1",
                    "receipt_digest": "a" * 64,
                },
            }
        )
        + "\n\n"
        + "id: 1+1\n"
        + "event: event\n"
        + "data: "
        + json.dumps(
            {
                "event_id": "evt_BBBBBBBBBBBBBBBBBBBBBB",
                "run_id": RUN_ID,
                "sequence": 1,
                "kind": "approval.requested",
                "original_kind": "approval.requested",
                "usage": None,
                "observed_at": "2026-11-27T15:00:02Z",
                "approval_id": "apr-7",
                "payload": {
                    "approval_id": "apr-7",
                    "action_kind": "git_push",
                    "summary": "Push release branch",
                    "requested_by": {"actor_id": "planner", "actor_kind": "actor"},
                    "requested_at": "2026-11-27T15:00:02Z",
                    "timeout_ms": 30000,
                    "expires_at": "2026-11-27T15:00:32Z",
                    "revision": 1,
                    "status": "pending",
                },
            }
        )
        + "\n\n"
        + "id: 1.3+1\n"
        + "event: milestone\n"
        + "data: "
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
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )
        if (
            request.method == "POST"
            and request.url.path == f"/v1/runs/{RUN_ID}/approvals/apr-7/decide"
        ):
            payload = json.loads(request.content.decode("utf-8"))
            decided_requests.append(payload)
            return httpx.Response(
                200,
                json={
                    "operation_id": "op-elicit-1",
                    "run_id": RUN_ID,
                    "approval_id": "apr-7",
                    "decision": payload["decision"],
                    "status": "approved",
                    "revision": 2,
                    "state": "applied",
                },
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_updates.append((progress, total, message))

    async def on_log(params: LoggingMessageNotificationParams) -> None:
        log_messages.append(str(params.data))

    async def on_elicit(
        context: RequestContext[ClientSession, None],
        params: ElicitRequestParams,
    ) -> ElicitResult:
        del context
        elicited_prompts.append(params.message)
        return ElicitResult(action="accept", content={"decision": "approve", "reason": "lgtm"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(
            server,
            logging_callback=on_log,
            elicitation_callback=on_elicit,
        ) as session:
            result = await session.call_tool(
                "swarms_run_wait",
                {"run_id": RUN_ID, "timeout_seconds": 5},
                progress_callback=on_progress,
            )

    assert not result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    body = json.loads(content.text)
    assert body["outcome"] == "terminal"
    assert any(msg and "Objective accuracy: 75%" in msg for _, _, msg in progress_updates)
    assert any(f"Run {RUN_ID} terminal (state=completed)" in msg for msg in log_messages)
    assert any("Approval apr-7: decided (approve)" in msg for msg in log_messages)
    assert len(elicited_prompts) == 1
    assert "apr-7" in elicited_prompts[0]
    assert decided_requests == [{"decision": "approve", "reason": "lgtm", "expected_revision": 1}]


async def test_stdio_mcp_wait_degrades_gracefully_without_elicitation(
    contract: dict[str, object],
) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    progress_updates: list[tuple[float, float | None, str | None]] = []
    sse_body = (
        "id: 1+1\n"
        "event: event\n"
        + "data: "
        + json.dumps(
            {
                "event_id": EVT_ID,
                "run_id": RUN_ID,
                "sequence": 1,
                "kind": "approval.requested",
                "original_kind": "approval.requested",
                "usage": None,
                "observed_at": "2026-11-27T15:00:02Z",
                "approval_id": "apr-8",
                "payload": {
                    "approval_id": "apr-8",
                    "action_kind": "git_push",
                    "summary": "Push branch",
                    "requested_by": {"actor_id": "planner", "actor_kind": "actor"},
                    "requested_at": "2026-11-27T15:00:02Z",
                    "timeout_ms": 30000,
                    "expires_at": "2026-11-27T15:00:32Z",
                    "revision": 1,
                    "status": "pending",
                },
            }
        )
        + "\n\n"
        + "id: 1.3+1\n"
        + "event: milestone\n"
        + "data: "
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
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_updates.append((progress, total, message))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        # No elicitation_callback -> client does not advertise elicitation capability.
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "swarms_run_wait",
                {"run_id": RUN_ID, "timeout_seconds": 5},
                progress_callback=on_progress,
            )

    assert not result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text)["outcome"] == "terminal"
    assert any(msg and "Approval requested (apr-8)" in msg for _, _, msg in progress_updates)


async def test_stdio_mcp_wait_streams_run_milestone_progress_notifications(
    contract: dict[str, object],
) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    progress_updates: list[tuple[float, float | None, str | None]] = []
    milestones = [
        (
            "1.1",
            {
                "milestone": "execution_closed",
                "run_id": RUN_ID,
                "state": "completed",
                "execution_closed": True,
                "accounting_final": False,
                "cleanup_complete": False,
                "result_available": False,
                "failure": None,
                "final": False,
            },
        ),
        (
            "1.2",
            {
                "milestone": "accounting_final",
                "run_id": RUN_ID,
                "state": "completed",
                "execution_closed": True,
                "accounting_final": True,
                "cleanup_complete": False,
                "result_available": True,
                "failure": None,
                "final": False,
            },
        ),
        (
            "1.3",
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
            },
        ),
    ]
    sse_body = "".join(
        f"id: {frame_id}\nevent: milestone\ndata: {json.dumps(frame)}\n\n"
        for frame_id, frame in milestones
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/events":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=_run_snapshot_example(contract, state="completed"))
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_updates.append((progress, total, message))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "swarms_run_wait",
                {"run_id": RUN_ID, "timeout_seconds": 5},
                progress_callback=on_progress,
            )

    assert not result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text)["outcome"] == "terminal"
    assert (1.0, 3.0, "Milestone execution_closed (state=completed)") in progress_updates
    assert (2.0, 3.0, "Milestone accounting_final (state=completed)") in progress_updates
    assert (3.0, 3.0, "Milestone cleanup_complete (state=completed)") in progress_updates
