"""Typed pending-action, task/work-graph reads and the runtime-unavailable refusal.

All requests use httpx MockTransport. No model, provider or live mutation.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from synth_ai.core.errors import RuntimeUnavailableError, TransientServiceError
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.retry import RetryPolicy
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.common import ProjectId, SwarmId
from synth_ai.sdk.research.contracts.swarm_tasks import RetryKind, ReviewVerdict
from synth_ai.sdk.research.swarms import AsyncSwarmsAPI, SwarmsAPI

RUN = SwarmId("run-reads-fixture")
PROJECT = ProjectId("project-reads-fixture")
AT = "2026-10-06T12:00:00+00:00"


def status(**extra):
    body = {
        "schema_version": 1,
        "run_id": RUN,
        "project_id": PROJECT,
        "state": {"public_state": "running", "reason": None, "blocker_code": None},
        "liveness": {"phase": "running", "task_counts": {}, "participant_queue_counts": {}},
        "terminal": {
            "run_state": "running",
            "run_state_is_terminal": False,
            "contract_status": None,
            "contract_reason": None,
            "outcome": None,
        },
        "finalization": {
            "status": None,
            "proof_status": None,
            "proof_satisfied": False,
            "blocker_code": None,
            "reason": None,
        },
        "recovery": {
            "blocked": False,
            "codes": [],
            "incident_count": 0,
            "latest_recorded_at": None,
        },
        "progress": {"last_progress_at": None, "last_progress_kind": None},
        "issues": {"issues": [], "invariants": []},
        "failure": {"code": None, "source": None, "classification": None, "terminal": False},
        "freshness": {
            "projection_authority": "smr_run_status_projection.v1",
            "last_authoritative_update_at": AT,
            "generated_at": AT,
        },
    }
    body.update(extra)
    return body


QUESTION = {
    "action_id": "action_7",
    "execution_id": "execution_3",
    "question_text": "Which dataset?",
    "question_text_ref": {"artifact_id": "artifact_1"},
    "asked_at": AT,
}


def task(key, *, depends=(), retry_of=None, state="done", run=RUN):
    return {
        "task_id": f"id-{key}",
        "run_id": run,
        "org_id": "org-fixture",
        "project_id": PROJECT,
        "task_key": key,
        "kind": "worker",
        "public_task_state": state,
        "task_state": state,
        "depends_on_task_keys": list(depends),
        "retry_of": retry_of,
        "input": {},
        "agent_goal_assignment": None,
        "output": {},
        "diagnostics": {},
        "execution_owner": "worker",
        "task_dispatch": {},
        "worker_pool": None,
        "claimed_by": None,
        "lease_expires_at": None,
        "last_heartbeat_at": None,
        "created_at": AT,
        "started_at": None,
        "finished_at": None,
        "updated_at": AT,
    }


def event(event_id, key, state, minute, previous=None):
    return {
        "event_id": event_id,
        "event_kind": "task_updated",
        "display_kind": "task_updated",
        "task_id": f"id-{key}",
        "task_key": key,
        "actor_id": None,
        "actor_label": None,
        "participant_role": "reviewer",
        "previous_event_state": previous,
        "event_state": state,
        "summary": "opaque text is never interpreted",
        "source": "task",
        "source_row_kind": "logical_timeline_node",
        "source_row_id": event_id,
        "payload": {},
        "occurred_at": f"2026-10-06T12:{minute:02d}:00+00:00",
    }


TASKS = [
    task("build"),
    task("build-retry", retry_of="id-build"),
    task("report", depends=["build"]),
]
EVENTS = {
    "schema_version": "smr_run_task_events.v1",
    "project_id": PROJECT,
    "run_id": RUN,
    "generated_at": AT,
    "events": [
        event("e1", "build", "running", 1),
        event("e2", "build", "review_required", 2),
        event("e3", "build", "repair_required", 3),
        event("e4", "build", "running", 4),
        event("e5", "build", "review_required", 5),
        event("e6", "build", "done", 6, previous="review_required"),
        event("e7", "report", "done", 7),
    ],
    "cursor": {"order": "occurred_at_asc_event_id_asc", "next": None},
}


def routes(request):
    path = request.url.path
    if path.endswith("/status"):
        return httpx.Response(200, json=status(pending_questions=[QUESTION]))
    if path.endswith("/tasks"):
        return httpx.Response(200, json=TASKS)
    if path.endswith("/task-events"):
        return httpx.Response(200, json=EVENTS)
    raise AssertionError(f"unexpected {request.method} {path}")


def transport(handler, attempts=3):
    result = HttpTransport("https://backend.test", {}, retry_policy=RetryPolicy(attempts, 0, 0))
    result.client.close()
    result.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://backend.test"
    )
    return result


def test_pending_actions_carry_the_action_id_answer_action_needs():
    requests = []

    def handler(request):
        requests.append(request)
        return routes(request)

    wire = transport(handler)
    try:
        (pending,) = SwarmsAPI(wire).pending_actions(RUN)
        assert (pending.action_id, pending.execution_id) == ("action_7", "execution_3")
        assert pending.question_text == "Which dataset?"
        assert pending.question_text_ref == {"artifact_id": "artifact_1"}
        assert pending.asked_at is not None
        assert [(r.method, r.url.path) for r in requests] == [("GET", f"/smr/runs/{RUN}/status")]
    finally:
        wire.close()


def test_status_without_pending_questions_and_drifted_rows():
    wire = transport(lambda request: httpx.Response(200, json=status()))
    try:
        assert SwarmsAPI(wire).status(RUN).pending_actions == ()
    finally:
        wire.close()
    for bad in [{"action_id": "a"}, {**QUESTION, "extra": 1}, {**QUESTION, "action_id": " "}]:
        wire = transport(
            lambda request, bad=bad: httpx.Response(200, json=status(pending_questions=[bad]))
        )
        try:
            with pytest.raises(ValueError):
                SwarmsAPI(wire).pending_actions(RUN)
        finally:
            wire.close()


def test_status_round_trips_pending_actions():
    from synth_ai.sdk.research.contracts.status import SwarmStatus

    parsed = SwarmStatus.from_wire(status(pending_questions=[QUESTION]))
    assert SwarmStatus.from_wire(parsed.to_wire()) == parsed


def test_work_graph_types_dependencies_verdicts_and_retries():
    requests = []

    def handler(request):
        requests.append(request)
        return routes(request)

    wire = transport(handler)
    try:
        graph = SwarmsAPI(wire).work_graph(RUN, PROJECT, limit=200)
    finally:
        wire.close()
    assert [(r.url.path, dict(r.url.params)) for r in requests] == [
        (f"/smr/runs/{RUN}/tasks", {}),
        (f"/smr/projects/{PROJECT}/runs/{RUN}/task-events", {"limit": "200"}),
    ]
    assert [t.task_key for t in graph.dependents_of("build")] == ["report"]
    assert [(d.upstream_task_key, d.downstream_task_key) for d in graph.dependencies] == [
        ("build", "report")
    ]
    assert [(d.verdict, d.event_id) for d in graph.decisions_for("build")] == [
        (ReviewVerdict.REPAIR_REQUESTED, "e3"),
        (ReviewVerdict.ACCEPTED, "e6"),
    ]
    assert graph.decisions_for("report") == ()
    assert {(r.kind, r.task_ref, r.retry_of, r.event_id) for r in graph.retries_of("build")} == {
        (RetryKind.NEW_ATTEMPT, "build-retry", "id-build", None),
        (RetryKind.REPAIR, "build", None, "e4"),
    }
    assert graph.task("id-report").task_key == "report"


@pytest.mark.parametrize(
    "path,body",
    [
        ("/tasks", [task("x", run="foreign")]),
        ("/tasks", [{**task("x"), "unknown": 1}]),
        ("/tasks", [{**task("x"), "depends_on_task_keys": [""]}]),
        ("/task-events", {**EVENTS, "run_id": "foreign"}),
        ("/task-events", {**EVENTS, "schema_version": "v2"}),
        ("/task-events", {**EVENTS, "events": [{**EVENTS["events"][0], "extra": 1}]}),
    ],
)
def test_drifted_or_foreign_reads_refuse(path, body):
    def handler(request):
        if request.url.path.endswith(path):
            return httpx.Response(200, json=body)
        return routes(request)

    wire = transport(handler)
    try:
        with pytest.raises(ValueError):
            SwarmsAPI(wire).work_graph(RUN, PROJECT)
    finally:
        wire.close()


def test_limit_validation_precedes_any_network_request():
    def forbidden(request):
        raise AssertionError("invalid read reached the transport")

    wire = transport(forbidden)
    try:
        for limit in [0, 1001, True]:
            with pytest.raises(ValueError):
                SwarmsAPI(wire).task_events(RUN, PROJECT, limit=limit)
    finally:
        wire.close()


UNAVAILABLE = [
    (503, {"detail": {"error_code": "orchestra_unavailable", "retryable": True}}),
    (409, {"detail": {"error_code": "orchestra_unavailable", "message": "blocked"}}),
    (503, {"detail": {"error": "swarm_status_orchestra_unavailable", "retryable": True}}),
]


@pytest.mark.parametrize("code,body", UNAVAILABLE)
def test_runtime_unavailable_is_typed_and_never_retried(code, body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(code, json=body)

    wire = transport(handler)
    try:
        api = SwarmsAPI(wire)
        with pytest.raises(RuntimeUnavailableError) as raised:
            api.steer(RUN, "message", idempotency_key="stable-key")
        with pytest.raises(RuntimeUnavailableError):
            api.status(RUN)
    finally:
        wire.close()
    assert len(calls) == 2
    assert raised.value.status == code
    assert not isinstance(raised.value, TransientServiceError)


def test_untyped_503_remains_a_retried_transient_failure():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": {"error_code": "orchestra_in_doubt"}})

    wire = transport(handler, attempts=2)
    try:
        with pytest.raises(TransientServiceError):
            SwarmsAPI(wire).status(RUN)
    finally:
        wire.close()
    assert len(calls) == 2


def test_async_reads_and_refusal_match_sync():
    async def scenario():
        calls = []

        def handler(request):
            calls.append(request)
            if request.url.path.endswith("/steer"):
                return httpx.Response(UNAVAILABLE[0][0], json=UNAVAILABLE[0][1])
            return routes(request)

        wire = AsyncHttpTransport("https://backend.test", {}, retry_policy=RetryPolicy(3, 0, 0))
        await wire.client.aclose()
        wire.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://backend.test"
        )
        try:
            api = AsyncSwarmsAPI(wire)
            assert [p.action_id for p in await api.pending_actions(RUN)] == ["action_7"]
            assert len(await api.tasks(RUN)) == 3
            graph = await api.work_graph(RUN, PROJECT)
            assert graph.decisions_for("build")[-1].verdict is ReviewVerdict.ACCEPTED
            with pytest.raises(RuntimeUnavailableError):
                await api.steer(RUN, "message", idempotency_key="key")
            assert sum(r.url.path.endswith("/steer") for r in calls) == 1
        finally:
            await wire.close()

    asyncio.run(scenario())


def test_catalog_mirrors_backend_authored_openapi():
    from synth_ai.sdk.research.operations import research_operation

    schema = json.loads((Path(__file__).parents[1] / "openapi/research-v1.json").read_text())
    for name, path in [
        ("list_run_tasks", "/smr/runs/{run_id}/tasks"),
        ("list_project_run_task_events", "/smr/projects/{project_id}/runs/{run_id}/task-events"),
        ("retrieve_swarm_status", "/smr/runs/{run_id}/status"),
    ]:
        assert schema["paths"][path]["get"]["operationId"] == name
        operation = research_operation(name)
        assert operation.path_template == path and operation.idempotent and not operation.mutation
    schemas = schema["components"]["schemas"]
    assert "pending_questions" in schemas["SmrSwarmStatusResponse"]["properties"]
    assert set(schemas["SmrSwarmStatusPendingQuestionResponse"]["properties"]) == set(QUESTION)
