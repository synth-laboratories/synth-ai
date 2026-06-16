import json

import synth_ai.managed_research
import synth_ai.managed_research.models as public_models
import synth_ai.managed_research.sdk as public_sdk
import pytest
from synth_ai.managed_research import (
    ActorResourceCapability,
    ManagedResearchProject,
    ManagedResearchRun,
    ManagedResearchRunControlEnqueueStatus,
    ManagedResearchRunControlError,
    Provider,
    SmrApiError,
    SmrControlClient,
    SmrHostKind,
    SmrWorkMode,
)
from synth_ai.managed_research.models.run_control import (
    ManagedResearchRunControlAck,
    RunLifecycleControlErrorCode,
)
from synth_ai.managed_research.models.runtime_intent import (
    RuntimeIntent,
    RuntimeIntentReceipt,
    RuntimeIntentStatus,
    RuntimeIntentView,
)
from synth_ai.managed_research.sdk import (
    CredentialsAPI,
    ExportsAPI,
    GithubAPI,
    ProgressAPI,
    ProjectsAPI,
    RunsAPI,
    UsageAPI,
    WorkspaceInputsAPI,
)


def test_legacy_provider_enums_are_not_public_launch_exports() -> None:
    for namespace in (synth_ai.managed_research, public_models, public_sdk):
        assert not hasattr(namespace, "SmrInferenceProvider")
        assert not hasattr(namespace, "SmrResourceProvider")

    assert hasattr(synth_ai.managed_research, "Provider")
    assert hasattr(synth_ai.managed_research, "ProviderBinding")
    assert hasattr(synth_ai.managed_research, "UsageLimit")
    assert hasattr(synth_ai.managed_research, "ActorResourceCapability")


def test_namespace_properties_are_stable() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    assert isinstance(client.projects, ProjectsAPI)
    assert isinstance(client.runs, RunsAPI)
    assert isinstance(client.workspace_inputs, WorkspaceInputsAPI)
    assert isinstance(client.progress, ProgressAPI)
    assert isinstance(client.usage, UsageAPI)
    assert isinstance(client.github, GithubAPI)
    assert isinstance(client.credentials, CredentialsAPI)
    assert isinstance(client.exports, ExportsAPI)
    assert callable(client.projects.get_agent_models)
    assert callable(client.projects.get_capacity_lane_preview)
    assert callable(client.projects.get_run_start_blockers)
    assert callable(client.projects.set_provider_key)
    assert callable(client.projects.get_provider_key_status)
    assert callable(client.projects.pause)
    assert callable(client.projects.resume)
    assert callable(client.projects.archive)
    assert callable(client.projects.unarchive)
    assert callable(client.projects.get_notes)
    assert callable(client.projects.set_notes)
    assert callable(client.projects.append_notes)
    assert client.projects is client.projects
    assert client.runs is client.runs
    assert client.workspace_inputs is client.workspace_inputs
    assert client.progress is client.progress
    assert client.usage is client.usage
    assert client.github is client.github
    assert client.credentials is client.credentials
    assert client.exports is client.exports
    assert callable(client.runs.get_logical_timeline)
    assert callable(client.runs.branch_from_checkpoint)
    assert callable(client.github.start_oauth)
    project = client.project("project-123")
    assert callable(project.repos.list)
    assert callable(project.files.list)
    assert callable(project.datasets.list)
    assert callable(project.prs.list)
    assert callable(project.models.list)
    assert callable(project.outputs.list)
    assert callable(project.readiness)

    client.close()


def test_projects_namespace_fetches_agent_model_catalog(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str]] = []
    payload = {
        "version": 1,
        "models": [
            {
                "id": "x-ai/grok-4.1-fast",
                "provider": "openrouter",
                "harnesses": ["opencode_sdk"],
                "usage_required": True,
                "pricing_present": True,
            }
        ],
    }

    def _request(method: str, path: str, **_kwargs):
        calls.append((method, path))
        return payload

    monkeypatch.setattr(client, "_request_json", _request)

    assert client.get_agent_models() == payload
    assert client.projects.get_agent_models() == payload
    assert calls == [("GET", "/smr/agent-models"), ("GET", "/smr/agent-models")]
    client.close()


def test_projects_namespace_returns_typed_models(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    monkeypatch.setattr(
        client,
        "list_projects",
        lambda **kwargs: [
            {
                "project_id": "proj_123",
                "org_id": "org_123",
                "name": "Alpha",
                "timezone": "UTC",
                "schedule": {},
                "budgets": {},
                "key_policy": {},
                "integrations": {},
                "project_repo": {},
                "repos": ["github.com/synth/example"],
                "onboarding_state": {},
                "research": {},
                "synth_ai": {},
                "policy": {},
                "trial_matrix": {},
                "execution": {},
                "created_at": "2026-04-15T12:00:00Z",
                "updated_at": "2026-04-15T12:30:00Z",
                "archived": False,
            }
        ],
    )
    monkeypatch.setattr(
        client,
        "get_project",
        lambda project_id: {
            "project_id": project_id,
            "org_id": "org_123",
            "name": "Alpha",
            "timezone": "UTC",
            "schedule": {},
            "budgets": {},
            "key_policy": {},
            "integrations": {},
            "project_repo": {},
            "repos": [],
            "onboarding_state": {},
            "research": {},
            "synth_ai": {},
            "policy": {},
            "trial_matrix": {},
            "execution": {},
            "created_at": "2026-04-15T12:00:00Z",
            "updated_at": "2026-04-15T12:30:00Z",
            "archived": False,
        },
    )

    projects = client.projects.list()
    project = client.projects.get("proj_123")

    assert isinstance(projects[0], ManagedResearchProject)
    assert isinstance(project, ManagedResearchProject)
    assert project.project_id == "proj_123"
    client.close()


def test_runs_namespace_get_returns_typed_run(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    monkeypatch.setattr(
        client,
        "get_run",
        lambda run_id, **kwargs: {
            "run_id": run_id,
            "project_id": kwargs.get("project_id") or "proj_123",
            "state": "paused",
            "public_state": "paused",
            "live_phase": "waiting",
            "state_authority": "backend_public_run_state_projection.v1",
            "host_kind": "daytona",
            "resolved_host_kind": "daytona",
            "work_mode": "directed_effort",
            "resolved_work_mode": "directed_effort",
            "providers": [{"provider": "openrouter"}],
            "capabilities": ["inference"],
            "limit": {"max_spend_usd": 5.0},
        },
    )

    run = client.runs.get("run_123", project_id="proj_123")

    assert isinstance(run, ManagedResearchRun)
    assert run.run_id == "run_123"
    assert run.host_kind is SmrHostKind.DAYTONA
    assert run.resolved_host_kind is SmrHostKind.DAYTONA
    assert run.work_mode is SmrWorkMode.DIRECTED_EFFORT
    assert run.resolved_work_mode is SmrWorkMode.DIRECTED_EFFORT
    assert run.providers[0].provider is Provider.OPENROUTER
    assert run.capabilities == frozenset({ActorResourceCapability.INFERENCE})
    assert run.limit is not None
    assert run.limit.max_spend_usd == 5.0
    client.close()


def test_run_handle_pause_resume_stop_are_symmetric(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, str | None]] = []

    def _ack(run_id: str, *, project_id: str | None = None) -> dict[str, object]:
        calls.append(("control", run_id, project_id))
        return {
            "run_id": run_id,
            "project_id": project_id,
            "state": "paused",
            "public_state": "paused",
            "control_intent_id": "message:run_123:smr_runtime_control:1",
            "control_intent_ack_at": "2026-04-15T12:00:00Z",
            "enqueue_status": "accepted",
        }

    monkeypatch.setattr(client, "pause_run", _ack)
    monkeypatch.setattr(client, "resume_run", _ack)
    monkeypatch.setattr(client, "stop_run", _ack)

    handle = client.run("proj_123", "run_123")

    paused = handle.pause()
    resumed = handle.resume()
    stopped = handle.stop()

    assert paused.enqueue_status is ManagedResearchRunControlEnqueueStatus.ACCEPTED
    assert resumed.enqueue_status is ManagedResearchRunControlEnqueueStatus.ACCEPTED
    assert stopped.enqueue_status is ManagedResearchRunControlEnqueueStatus.ACCEPTED
    assert calls == [
        ("control", "run_123", "proj_123"),
        ("control", "run_123", "proj_123"),
        ("control", "run_123", "proj_123"),
    ]
    client.close()


def test_runs_namespace_uses_public_runtime_message_listing(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fail_private(*args, **kwargs):
        raise AssertionError("RunsAPI should not use _list_runtime_messages")

    def list_public(run_id: str, **kwargs):
        return [{"run_id": run_id, "status": kwargs.get("status") or "queued"}]

    monkeypatch.setattr(client, "_list_runtime_messages", fail_private)
    monkeypatch.setattr(client, "list_runtime_messages", list_public)

    messages = client.runs.list_runtime_messages("run_123", status="queued")

    assert messages == [{"run_id": "run_123", "status": "queued"}]
    client.close()


def test_runs_namespace_submits_typed_runtime_intent(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, object]]] = []

    def _request(method: str, path: str, **kwargs):
        calls.append((method, path, kwargs))
        return {
            "runtime_intent_id": "message:run_123:smr_runtime_control:4",
            "runtime_intent_status": "queued",
            "runtime_intent_ack_at": "2026-04-19T17:00:00Z",
            "run_id": "run_123",
            "intent_kind": "answer_question",
            "mode": "queue",
        }

    monkeypatch.setattr(client, "_request_json", _request)

    receipt = client.runs.submit_intent(
        "run_123",
        RuntimeIntent.answer_question(
            question_id="question-1",
            user_id="user-1",
            response_text="Proceed.",
        ),
        project_id="proj_123",
        body="Proceed.",
        causation_id="message:run_123:question:1",
    )

    assert isinstance(receipt, RuntimeIntentReceipt)
    assert receipt.runtime_intent_status is RuntimeIntentStatus.QUEUED
    assert receipt.runtime_intent_id == "message:run_123:smr_runtime_control:4"
    assert calls == [
        (
            "POST",
            "/smr/projects/proj_123/runs/run_123/runtime/intents",
            {
                "json_body": {
                    "intent": {
                        "kind": "answer_question",
                        "payload": {
                            "question_id": "question-1",
                            "user_id": "user-1",
                            "response_text": "Proceed.",
                            "requested_by_role": "human",
                        },
                    },
                    "mode": "queue",
                    "body": "Proceed.",
                    "causation_id": "message:run_123:question:1",
                }
            },
        )
    ]
    client.close()


def test_runs_namespace_lists_and_gets_runtime_intents(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, object]]] = []
    row = {
        "runtime_intent_id": "message:run_123:smr_runtime_control:5",
        "runtime_intent_status": "ignored",
        "runtime_intent_ack_at": "2026-04-19T17:00:00Z",
        "run_id": "run_123",
        "intent_kind": "answer_question",
        "mode": "queue",
        "message_id": "message:run_123:smr_runtime_control:5",
        "seq": 5,
        "action": "smr.intent.answer_question",
        "topic": "smr.intent",
        "causation_id": "message:run_123:question:1",
        "sender": "user:alice",
        "target": "role:system",
        "body": "Proceed.",
        "payload": {"question_id": "question-1"},
        "requested_by": "user:alice",
        "requested_by_role": "human",
        "resolved_at": "2026-04-19T17:01:00Z",
        "error_code": "run_terminal",
        "error_detail": "Run became terminal.",
        "retryable": False,
        "applied_mode": "noop",
    }

    def _request(method: str, path: str, **kwargs):
        calls.append((method, path, kwargs))
        if method == "GET" and path.endswith("/runtime/intents"):
            return [row]
        return row

    monkeypatch.setattr(client, "_request_json", _request)

    rows = client.runs.intents(
        "run_123",
        project_id="proj_123",
        status="ignored",
        limit=10,
    )
    one = client.runs.intent(
        "run_123",
        "message:run_123:smr_runtime_control:5",
        project_id="proj_123",
    )

    assert isinstance(rows[0], RuntimeIntentView)
    assert rows[0].runtime_intent_status is RuntimeIntentStatus.IGNORED
    assert rows[0].causation_id == "message:run_123:question:1"
    assert one.runtime_intent_id == "message:run_123:smr_runtime_control:5"
    assert calls == [
        (
            "GET",
            "/smr/projects/proj_123/runs/run_123/runtime/intents",
            {"params": {"status": "ignored", "limit": 10}},
        ),
        (
            "GET",
            "/smr/projects/proj_123/runs/run_123/runtime/intents/message:run_123:smr_runtime_control:5",
            {},
        ),
    ]
    client.close()


def test_run_control_ack_roundtrips_noop_status() -> None:
    ack = ManagedResearchRunControlAck.from_wire(
            {
                "run_id": "run_123",
                "project_id": "proj_123",
                "state": "paused",
                "public_state": "paused",
                "control_intent_id": "message:run_123:smr_runtime_control:2",
            "control_intent_ack_at": "2026-04-15T12:00:00Z",
            "enqueue_status": "noop",
        }
    )

    assert ack.enqueue_status is ManagedResearchRunControlEnqueueStatus.NOOP
    assert ack.control_intent_id == "message:run_123:smr_runtime_control:2"
    assert ack.control_intent_ack_at is not None


def test_run_control_ack_roundtrips_terminal_sync_with_null_intent() -> None:
    ack = ManagedResearchRunControlAck.from_wire(
            {
                "run_id": "run_123",
                "project_id": "proj_123",
                "state": "stopped",
                "public_state": "stopped",
                "control_intent_id": None,
            "control_intent_ack_at": None,
            "enqueue_status": "terminal_sync",
        }
    )

    assert ack.enqueue_status is ManagedResearchRunControlEnqueueStatus.TERMINAL_SYNC
    assert ack.control_intent_id is None
    assert ack.control_intent_ack_at is None


@pytest.mark.parametrize(
    "code",
    [
        RunLifecycleControlErrorCode.ALREADY_IN_STATE,
        RunLifecycleControlErrorCode.TERMINAL_RUN,
        RunLifecycleControlErrorCode.RUNTIME_NOT_LIVE,
        RunLifecycleControlErrorCode.RUN_NOT_FOUND,
    ],
)
def test_stop_run_maps_409_to_typed_control_error(monkeypatch, code) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    detail = {
        "error_code": code.value,
        "message": f"rejected: {code.value}",
        "retryable": code is RunLifecycleControlErrorCode.RUNTIME_NOT_LIVE,
        "current_state": "running",
        "run_id": "run_123",
    }
    body = json.dumps({"detail": detail})

    def _raise(*_args, **_kwargs):
        raise SmrApiError(
            f"POST failed with 409: {detail['message']}",
            status_code=409,
            response_text=body,
        )

    monkeypatch.setattr(client, "_request_json", _raise)

    with pytest.raises(ManagedResearchRunControlError) as exc_info:
        client.stop_run("run_123", project_id="proj_123")

    err = exc_info.value
    assert err.error_code is code
    assert err.retryable is (code is RunLifecycleControlErrorCode.RUNTIME_NOT_LIVE)
    assert err.current_state == "running"
    assert err.run_id == "run_123"
    assert err.status_code == 409
    assert err.detail == detail

    client.close()


def test_pause_run_maps_409_to_typed_control_error(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    detail = {
        "error_code": "already_in_state",
        "message": "already paused",
        "retryable": False,
        "current_state": "paused",
        "run_id": "run_abc",
    }

    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *a, **k: (_ for _ in ()).throw(
            SmrApiError(
                "409",
                status_code=409,
                response_text=json.dumps({"detail": detail}),
            )
        ),
    )

    with pytest.raises(ManagedResearchRunControlError) as exc_info:
        client.pause_run("run_abc", project_id="proj_123")

    assert exc_info.value.error_code is RunLifecycleControlErrorCode.ALREADY_IN_STATE
    client.close()


def test_resume_run_maps_409_to_typed_control_error(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    detail = {
        "error_code": "terminal_run",
        "message": "run is terminal",
        "retryable": False,
        "current_state": "completed",
        "run_id": "run_xyz",
    }

    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *a, **k: (_ for _ in ()).throw(
            SmrApiError(
                "409",
                status_code=409,
                response_text=json.dumps({"detail": detail}),
            )
        ),
    )

    with pytest.raises(ManagedResearchRunControlError) as exc_info:
        client.resume_run("run_xyz")

    assert exc_info.value.error_code is RunLifecycleControlErrorCode.TERMINAL_RUN
    client.close()


def test_run_control_409_with_malformed_body_raises_value_error(monkeypatch) -> None:
    """Contract drift (HTTP 409 without documented detail shape) must not be
    collapsed into a generic ``SmrApiError``; surface it as ``ValueError`` so
    callers can distinguish contract violations from recognised rejections.
    """

    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *a, **k: (_ for _ in ()).throw(
            SmrApiError(
                "409",
                status_code=409,
                # Missing `detail` mapping entirely — contract drift.
                response_text=json.dumps({"error": "something went wrong"}),
            )
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        client.stop_run("run_123")

    # Informative: the message must mention the contract keys we expected.
    assert "detail" in str(exc_info.value)
    client.close()


def test_run_control_409_with_unknown_error_code_raises_value_error(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    detail = {
        "error_code": "some_unexpected_code",
        "message": "drift",
        "retryable": False,
        "current_state": "running",
        "run_id": "run_123",
    }
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *a, **k: (_ for _ in ()).throw(
            SmrApiError(
                "409",
                status_code=409,
                response_text=json.dumps({"detail": detail}),
            )
        ),
    )

    with pytest.raises(ValueError):
        client.pause_run("run_123")

    client.close()


def test_mcp_tool_pause_run_serializes_enqueue_status_as_string(monkeypatch) -> None:
    """MCP responses go through ``json.dumps``; the ack's enum field must
    serialise to its string value, not an enum repr.
    """

    from synth_ai.managed_research.mcp.server import ManagedResearchMcpServer

    server = ManagedResearchMcpServer()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        class _Runs:
            def pause(self, run_id: str, *, project_id: str | None = None):
                return ManagedResearchRunControlAck.from_wire(
                        {
                            "run_id": run_id,
                            "project_id": project_id,
                            "state": "paused",
                            "public_state": "paused",
                            "control_intent_id": "message:run_x:smr_runtime_control:1",
                        "control_intent_ack_at": "2026-04-15T12:00:00Z",
                        "enqueue_status": "accepted",
                    }
                )

        runs = _Runs()

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    result = server._tool_pause_run({"run_id": "run_x", "project_id": "proj_1"})
    # Round-trip through JSON the way ``_write_message`` does.
    encoded = json.loads(json.dumps(result, default=str))
    assert encoded["enqueue_status"] == "accepted"
    assert isinstance(encoded["enqueue_status"], str)
