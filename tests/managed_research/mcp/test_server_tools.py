import pytest
from synth_ai.managed_research.errors import SmrFundingLaneInvariantError, SmrLimitExceededError
from synth_ai.managed_research.mcp.server import ManagedResearchMcpServer, RpcError
from synth_ai.managed_research.models.runtime_intent import (
    RuntimeIntentReceipt,
    RuntimeIntentView,
)


def test_rewritten_mcp_server_exposes_canonical_launch_and_workspace_tools() -> None:
    server = ManagedResearchMcpServer()
    names = set(server.available_tool_names())

    assert {
        "smr_append_project_notes",
        "smr_archive_project",
        "smr_attach_source_repo",
        "smr_create_runnable_project",
        "smr_get_capacity_lane_preview",
        "smr_get_capabilities",
        "smr_get_billing_entitlements",
        "smr_get_project_entitlement",
        "smr_get_project_git",
        "smr_get_project_notes",
        "smr_get_project_setup",
        "smr_get_project_usage",
        "smr_get_provider_key_status",
        "smr_download_workspace_archive",
        "smr_patch_project",
        "smr_pause_project",
        "smr_prepare_project_setup",
        "smr_rename_project",
        "smr_resume_project",
        "smr_set_project_notes",
        "smr_get_launch_preflight",
        "smr_get_run",
        "smr_get_run_logical_timeline",
        "smr_get_run_usage",
        "smr_unarchive_project",
        "smr_get_workspace_download_url",
        "smr_get_workspace_inputs",
        "smr_branch_run_from_checkpoint",
        "smr_list_active_runs",
        "smr_list_run_checkpoints",
        "smr_list_run_log_archives",
        "smr_list_run_questions",
        "smr_restore_run_checkpoint",
        "smr_runtime_intents",
        "smr_set_provider_key",
        "smr_trigger_run",
        "smr_upload_workspace_files",
    }.issubset(names)

    assert "smr_get_run_progress" not in names
    assert "smr_get_semantic_progress" not in names
    assert "smr_upload_starting_data" not in names
    assert "smr_get_starting_data_upload_urls" not in names
    assert "smr_trigger_data_factory" not in names
    assert "smr_data_factory_finalize" not in names
    assert "smr_data_factory_publish" not in names


def test_noun_surfaces_keep_sdk_mcp_suffix_parity() -> None:
    server = ManagedResearchMcpServer()
    names = set(server.available_tool_names())

    expected = {
        "github_status": "smr_setup_github_status",
        "github_start_oauth": "smr_setup_github_start_oauth",
        "github_list_repos": "smr_setup_github_list_repos",
        "github_disconnect": "smr_setup_github_disconnect",
        "exports_list_targets": "smr_setup_exports_list_targets",
        "exports_create_target": "smr_setup_exports_create_target",
        "repos_list": "smr_work_repos_list",
        "repos_attach": "smr_work_repos_attach",
        "repos_detach": "smr_work_repos_detach",
        "datasets_list": "smr_work_datasets_list",
        "datasets_upload": "smr_work_datasets_upload",
        "datasets_download": "smr_work_datasets_download",
        "files_list": "smr_work_files_list",
        "files_upload": "smr_work_files_upload",
        "prs_list": "smr_results_prs_list",
        "prs_get": "smr_results_prs_get",
        "models_list": "smr_results_models_list",
        "models_get": "smr_results_models_get",
        "models_download": "smr_results_models_download",
        "models_export": "smr_results_models_export",
        "readiness": "smr_status_readiness",
    }

    missing = {
        suffix: tool_name for suffix, tool_name in expected.items() if tool_name not in names
    }
    assert missing == {}


def test_github_oauth_start_tool_routes_to_canonical_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def start_github_oauth(self, *, redirect_uri=None):
            captured["redirect_uri"] = redirect_uri
            return {
                "authorize_url": "https://github.com/apps/synth/install",
                "state": "org-123",
            }

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    response = server._tool_setup_github_start_oauth(
        {"redirect_uri": "https://app.usesynth.ai/integrations/github"}
    )

    assert response["authorize_url"] == "https://github.com/apps/synth/install"
    assert captured["redirect_uri"] == "https://app.usesynth.ai/integrations/github"


def test_branch_tool_validates_checkpoint_reference() -> None:
    server = ManagedResearchMcpServer()

    with pytest.raises(ValueError, match="exactly one of checkpoint_id"):
        server._tool_branch_run_from_checkpoint({"run_id": "run_123"})


def test_branch_tool_routes_to_canonical_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def branch_run_from_checkpoint(self, run_id, **kwargs):
            captured["run_id"] = run_id
            captured["kwargs"] = kwargs
            return {
                "accepted": True,
                "parent_run_id": "run_parent",
                "child_run_id": "run_child",
                "source_checkpoint_id": "ckpt_123",
                "branch_message_id": "msg_123",
                "created_at": "2026-04-15T12:00:00Z",
            }

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    response = server._tool_branch_run_from_checkpoint(
        {
            "project_id": "proj_123",
            "run_id": "run_123",
            "checkpoint_id": "ckpt_123",
            "mode": "with_message",
            "message": "Try a different branch.",
            "title": "Alternative path",
        }
    )

    assert response["child_run_id"] == "run_child"
    assert captured["run_id"] == "run_123"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["project_id"] == "proj_123"
    assert kwargs["checkpoint_id"] == "ckpt_123"
    assert kwargs["checkpoint_record_id"] is None
    assert kwargs["checkpoint_uri"] is None
    assert getattr(kwargs["mode"], "value", None) == "with_message"
    assert kwargs["message"] == "Try a different branch."
    assert kwargs["reason"] is None
    assert kwargs["title"] == "Alternative path"
    assert kwargs["source_node_id"] is None


def test_project_usage_tool_routes_to_canonical_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def get_project_usage(self, project_id):
            captured["project_id"] = project_id
            return {
                "project_id": project_id,
                "month_to_date": {},
                "last_7_days": {},
                "per_run": [],
                "budgets": {},
            }

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    response = server._tool_get_project_usage(
        {
            "project_id": "proj_123",
        }
    )

    assert response["project_id"] == "proj_123"
    assert captured["project_id"] == "proj_123"


def test_health_check_surfaces_api_key_resolution_failure(monkeypatch) -> None:
    server = ManagedResearchMcpServer()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def get_capabilities(self) -> dict[str, str]:
            return {"version": "test"}

    monkeypatch.setattr(
        "synth_ai.managed_research.mcp.server.get_api_key",
        lambda required=False: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    out = server._tool_health_check({})

    assert out["ok"] is True
    assert out["checks"]["api_key"] == {
        "status": "fail",
        "configured": False,
        "message": "bad config",
    }


def test_trigger_run_returns_structured_payload_on_limit_exceeded(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    detail = {
        "error_code": "smr_limit_exceeded",
        "resource_id": "agent_daytona",
        "window": "daily",
    }

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def trigger_run(self, project_id: str, **kwargs):
            raise SmrLimitExceededError(
                "limit hit",
                status_code=429,
                response_text="{}",
                detail=detail,
            )

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    with pytest.raises(RpcError) as exc_info:
        server._tool_trigger_run(
            {
                "project_id": "proj_1",
                "host_kind": "daytona",
                "work_mode": "directed_effort",
                "providers": [{"provider": "openrouter"}],
            },
        )
    assert exc_info.value.data == {
        "error": "smr_limit_exceeded",
        "detail": detail,
        "message": "limit hit",
        "http_status": 429,
    }


def test_trigger_run_returns_structured_payload_on_routing_invariant(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    detail = {
        "error_code": "smr_free_tier_routing_violation",
        "invariant": "ga_free_must_not_use_synth_codex_pool",
    }

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def trigger_run(self, project_id: str, **kwargs):
            raise SmrFundingLaneInvariantError(
                "routing",
                status_code=409,
                response_text="{}",
                detail=detail,
            )

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    with pytest.raises(RpcError) as exc_info:
        server._tool_trigger_run(
            {
                "project_id": "proj_1",
                "host_kind": "daytona",
                "work_mode": "directed_effort",
                "providers": [{"provider": "openrouter"}],
            },
        )
    assert exc_info.value.data == {
        "error": "smr_free_tier_routing_violation",
        "detail": detail,
        "message": "routing",
        "http_status": 409,
    }


def test_trigger_run_returns_structured_payload_on_insufficient_credits(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    detail = {
        "error_code": "smr_insufficient_credits",
        "message": "Insufficient credits to start a new run.",
    }

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def trigger_run(self, project_id: str, **kwargs):
            from synth_ai.managed_research.errors import SmrInsufficientCreditsError

            raise SmrInsufficientCreditsError(
                "no credits",
                status_code=402,
                response_text="{}",
                detail=detail,
            )

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    with pytest.raises(RpcError) as exc_info:
        server._tool_trigger_run(
            {
                "project_id": "proj_1",
                "host_kind": "daytona",
                "work_mode": "directed_effort",
                "providers": [{"provider": "openrouter"}],
            },
        )
    assert exc_info.value.data == {
        "error": "smr_insufficient_credits",
        "detail": detail,
        "message": "no credits",
        "http_status": 402,
    }


def test_get_workspace_download_url_delegates_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, str] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def get_workspace_download_url(self, project_id: str) -> dict[str, str]:
            captured["project_id"] = project_id
            return {"download_url": "https://s3.example/presigned", "commit_sha": "abc"}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    out = server._tool_get_workspace_download_url({"project_id": "pid_1"})
    assert out == {"download_url": "https://s3.example/presigned", "commit_sha": "abc"}
    assert captured["project_id"] == "pid_1"


def test_get_capacity_lane_preview_delegates_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, str] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def get_capacity_lane_preview(self, project_id: str) -> dict[str, str]:
            captured["project_id"] = project_id
            return {"resolved_lane": "openai_chatgpt_pool"}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    out = server._tool_get_capacity_lane_preview({"project_id": "pid_1"})
    assert out == {"resolved_lane": "openai_chatgpt_pool"}
    assert captured["project_id"] == "pid_1"


def test_project_patch_and_notes_tools_delegate_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def patch_project(
            self,
            project_id: str,
            payload: dict[str, object],
            **kwargs,
        ):
            assert kwargs == {"actor_model_assignments": None}
            captured["patch"] = (project_id, payload)
            return {"project_id": project_id, **payload}

        def rename_project(self, project_id: str, name: str):
            captured["rename"] = (project_id, name)
            return {"project_id": project_id, "name": name}

        def get_project_notes(self, project_id: str):
            captured["get_notes"] = project_id
            return {"project_id": project_id, "notes": "remember this"}

        def set_project_notes(self, project_id: str, notes: str):
            captured["set_notes"] = (project_id, notes)
            return {"project_id": project_id, "notes": notes}

        def append_project_notes(self, project_id: str, notes: str):
            captured["append_notes"] = (project_id, notes)
            return {"project_id": project_id, "notes": f"prefix\n{notes}"}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    assert server._tool_patch_project({"project_id": "pid_1", "config": {"name": "Renamed"}}) == {
        "project_id": "pid_1",
        "name": "Renamed",
    }
    assert server._tool_rename_project({"project_id": "pid_1", "name": "  Friendly name  "}) == {
        "project_id": "pid_1",
        "name": "Friendly name",
    }
    assert server._tool_get_project_notes({"project_id": "pid_1"}) == {
        "project_id": "pid_1",
        "notes": "remember this",
    }
    assert server._tool_set_project_notes({"project_id": "pid_1", "notes": "fresh notes"}) == {
        "project_id": "pid_1",
        "notes": "fresh notes",
    }
    assert server._tool_append_project_notes({"project_id": "pid_1", "notes": "delta"}) == {
        "project_id": "pid_1",
        "notes": "prefix\ndelta",
    }
    assert captured == {
        "patch": ("pid_1", {"name": "Renamed"}),
        "rename": ("pid_1", "Friendly name"),
        "get_notes": "pid_1",
        "set_notes": ("pid_1", "fresh notes"),
        "append_notes": ("pid_1", "delta"),
    }


def test_project_rename_tool_rejects_blank_name() -> None:
    server = ManagedResearchMcpServer()

    with pytest.raises(ValueError, match="'name' is required and must be a non-empty string"):
        server._tool_rename_project({"project_id": "pid_1", "name": "   "})


def test_create_project_rejects_non_object_config(monkeypatch) -> None:
    server = ManagedResearchMcpServer()

    class _UnusedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

    monkeypatch.setattr(server, "_client_from_args", lambda args: _UnusedClient())

    with pytest.raises(ValueError, match="'config' must be an object when provided"):
        server._tool_create_project({"name": "x", "config": "bad"})


def test_project_lifecycle_tools_delegate_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: list[tuple[str, str]] = []

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def pause_project(self, project_id: str):
            captured.append(("pause", project_id))
            return {"project_id": project_id, "state": "paused"}

        def resume_project(self, project_id: str):
            captured.append(("resume", project_id))
            return {"project_id": project_id, "state": "active"}

        def archive_project(self, project_id: str):
            captured.append(("archive", project_id))
            return {"project_id": project_id, "archived": True}

        def unarchive_project(self, project_id: str):
            captured.append(("unarchive", project_id))
            return {"project_id": project_id, "archived": False}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    assert server._tool_pause_project({"project_id": "pid_1"}) == {
        "project_id": "pid_1",
        "state": "paused",
    }
    assert server._tool_resume_project({"project_id": "pid_1"}) == {
        "project_id": "pid_1",
        "state": "active",
    }
    assert server._tool_archive_project({"project_id": "pid_1"}) == {
        "project_id": "pid_1",
        "archived": True,
    }
    assert server._tool_unarchive_project({"project_id": "pid_1"}) == {
        "project_id": "pid_1",
        "archived": False,
    }
    assert captured == [
        ("pause", "pid_1"),
        ("resume", "pid_1"),
        ("archive", "pid_1"),
        ("unarchive", "pid_1"),
    ]


def test_get_run_start_blockers_delegates_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def get_run_start_blockers(self, project_id: str, **kwargs):
            captured["project_id"] = project_id
            captured["kwargs"] = kwargs
            return {"clear_to_trigger": False, "blockers": [{"stage": "limits"}]}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    out = server._tool_get_run_start_blockers(
        {
            "project_id": "pid_1",
            "host_kind": "daytona",
            "work_mode": "directed_effort",
            "providers": [{"provider": "openrouter"}],
            "agent_model_params": {"reasoning_effort": "high"},
            "initial_runtime_messages": [{"body": "Check staging first.", "mode": "queue"}],
            "sandbox_override": {"image": "synth/smr:latest"},
            "run_policy": {"limits": {"total_cost_cents": 1800}},
        }
    )
    assert out == {"clear_to_trigger": False, "blockers": [{"stage": "limits"}]}
    assert captured["project_id"] == "pid_1"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host_kind"] == "daytona"
    assert kwargs["work_mode"] == "directed_effort"
    assert kwargs["providers"] == [{"provider": "openrouter"}]
    assert kwargs["agent_model_params"] == {"reasoning_effort": "high"}
    assert kwargs["initial_runtime_messages"] == [{"body": "Check staging first.", "mode": "queue"}]
    assert kwargs["sandbox_override"] == {"image": "synth/smr:latest"}
    run_policy = kwargs["run_policy"]
    assert getattr(getattr(run_policy, "limits", None), "total_cost_cents", None) == 1800


def test_runtime_intents_tool_delegates_to_runs_namespace(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeRuns:
        def submit_intent(self, run_id: str, intent, **kwargs):
            captured.append(
                (
                    "submit",
                    {"run_id": run_id, "intent": intent, **kwargs},
                )
            )
            return RuntimeIntentReceipt.from_wire(
                {
                    "runtime_intent_id": "message:run_123:smr_runtime_control:6",
                    "runtime_intent_status": "queued",
                    "runtime_intent_ack_at": "2026-04-19T17:00:00Z",
                    "run_id": run_id,
                    "intent_kind": "answer_question",
                    "mode": kwargs.get("mode") or "queue",
                }
            )

        def intents(self, run_id: str, **kwargs):
            captured.append(("list", {"run_id": run_id, **kwargs}))
            return [
                RuntimeIntentView.from_wire(
                    {
                        "runtime_intent_id": "message:run_123:smr_runtime_control:6",
                        "runtime_intent_status": "applied",
                        "runtime_intent_ack_at": "2026-04-19T17:00:00Z",
                        "run_id": run_id,
                        "intent_kind": "answer_question",
                        "mode": "queue",
                        "message_id": "message:run_123:smr_runtime_control:6",
                        "seq": 6,
                        "action": "smr.intent.answer_question",
                        "topic": "smr.intent",
                    }
                )
            ]

        def intent(self, run_id: str, runtime_intent_id: str, **kwargs):
            captured.append(
                (
                    "get",
                    {
                        "run_id": run_id,
                        "runtime_intent_id": runtime_intent_id,
                        **kwargs,
                    },
                )
            )
            return RuntimeIntentView.from_wire(
                {
                    "runtime_intent_id": runtime_intent_id,
                    "runtime_intent_status": "applied",
                    "runtime_intent_ack_at": "2026-04-19T17:00:00Z",
                    "run_id": run_id,
                    "intent_kind": "answer_question",
                    "mode": "queue",
                }
            )

    class _FakeClient:
        runs = _FakeRuns()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    submitted = server._tool_runtime_intents(
        {
            "operation": "submit",
            "project_id": "proj_123",
            "run_id": "run_123",
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
    )
    listed = server._tool_runtime_intents(
        {
            "operation": "list",
            "project_id": "proj_123",
            "run_id": "run_123",
            "status": "applied",
            "limit": 5,
        }
    )
    fetched = server._tool_runtime_intents(
        {
            "operation": "get",
            "project_id": "proj_123",
            "run_id": "run_123",
            "runtime_intent_id": "message:run_123:smr_runtime_control:6",
        }
    )

    assert submitted["runtime_intent_status"] == "queued"
    assert listed[0]["runtime_intent_status"] == "applied"
    assert fetched["runtime_intent_id"] == "message:run_123:smr_runtime_control:6"
    assert captured == [
        (
            "submit",
            {
                "run_id": "run_123",
                "intent": {
                    "kind": "answer_question",
                    "payload": {
                        "question_id": "question-1",
                        "user_id": "user-1",
                        "response_text": "Proceed.",
                        "requested_by_role": "human",
                    },
                },
                "project_id": "proj_123",
                "mode": "queue",
                "body": "Proceed.",
                "causation_id": "message:run_123:question:1",
            },
        ),
        (
            "list",
            {
                "run_id": "run_123",
                "project_id": "proj_123",
                "status": "applied",
                "limit": 5,
            },
        ),
        (
            "get",
            {
                "run_id": "run_123",
                "runtime_intent_id": "message:run_123:smr_runtime_control:6",
                "project_id": "proj_123",
            },
        ),
    ]


def test_trigger_run_delegates_initial_runtime_messages_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def trigger_run(self, project_id: str, **kwargs):
            captured["project_id"] = project_id
            captured["kwargs"] = kwargs
            return {"run_id": "run_123"}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    out = server._tool_trigger_run(
        {
            "project_id": "pid_1",
            "host_kind": "daytona",
            "work_mode": "directed_effort",
            "providers": [{"provider": "openrouter"}],
            "initial_runtime_messages": [
                {"body": "Start with the launch blocker.", "mode": "queue"}
            ],
            "run_policy": {"access": {"tool_providers": ["tinker"]}},
        }
    )

    assert out == {"run_id": "run_123"}
    assert captured["project_id"] == "pid_1"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host_kind"] == "daytona"
    assert kwargs["work_mode"] == "directed_effort"
    assert kwargs["providers"] == [{"provider": "openrouter"}]
    assert kwargs["initial_runtime_messages"] == [
        {"body": "Start with the launch blocker.", "mode": "queue"}
    ]
    run_policy = kwargs["run_policy"]
    tool_providers = getattr(getattr(run_policy, "access", None), "tool_providers", ())
    assert [getattr(item, "value", item) for item in tool_providers] == ["tinker"]


def test_provider_key_tools_delegate_to_client(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    captured: list[tuple[str, str, dict[str, object]]] = []

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def set_provider_key(self, project_id: str, **kwargs):
            captured.append(("set", project_id, kwargs))
            return {"configured": True}

        def get_provider_key_status(self, project_id: str, **kwargs):
            captured.append(("status", project_id, kwargs))
            return {"configured": False}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    assert server._tool_set_provider_key(
        {
            "project_id": "pid_1",
            "provider": "openrouter",
            "funding_source": "customer_byok",
            "api_key": "sk-test-key",
        }
    ) == {"configured": True}
    assert server._tool_get_provider_key_status(
        {
            "project_id": "pid_1",
            "provider": "openrouter",
            "funding_source": "customer_byok",
        }
    ) == {"configured": False}
    assert captured == [
        (
            "set",
            "pid_1",
            {
                "provider": "openrouter",
                "funding_source": "customer_byok",
                "api_key": "sk-test-key",
                "encrypted_key_b64": None,
            },
        ),
        (
            "status",
            "pid_1",
            {
                "provider": "openrouter",
                "funding_source": "customer_byok",
            },
        ),
    ]


def test_trigger_run_rejects_legacy_prompt_arg(monkeypatch) -> None:
    server = ManagedResearchMcpServer()

    class _UnusedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

    monkeypatch.setattr(server, "_client_from_args", lambda args: _UnusedClient())

    with pytest.raises(ValueError, match="The `prompt` field is no longer supported"):
        server._tool_trigger_run(
            {
                "project_id": "pid_1",
                "host_kind": "daytona",
                "work_mode": "directed_effort",
                "prompt": "Ship it.",
            }
        )


def test_get_run_start_blockers_rejects_legacy_prompt_arg(monkeypatch) -> None:
    server = ManagedResearchMcpServer()

    class _UnusedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

    monkeypatch.setattr(server, "_client_from_args", lambda args: _UnusedClient())

    with pytest.raises(ValueError, match="The `prompt` field is no longer supported"):
        server._tool_get_run_start_blockers(
            {
                "project_id": "pid_1",
                "host_kind": "daytona",
                "work_mode": "directed_effort",
                "prompt": "Ship it.",
            }
        )


def test_download_workspace_archive_delegates_to_client(monkeypatch, tmp_path) -> None:
    server = ManagedResearchMcpServer()
    out_file = tmp_path / "ws.tar.gz"

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type
            del exc
            del tb

        def download_workspace_archive(self, project_id: str, output_path: str, **kwargs):
            assert project_id == "pid_1"
            assert str(out_file) in output_path or output_path.endswith("ws.tar.gz")
            assert kwargs.get("timeout_seconds") == 120
            return {"output_path": str(out_file), "bytes_written": 3, "commit_sha": "x"}

    monkeypatch.setattr(server, "_client_from_args", lambda args: _FakeClient())

    result = server._tool_download_workspace_archive(
        {
            "project_id": "pid_1",
            "output_path": str(out_file),
            "timeout_seconds": 120,
        }
    )
    assert result["bytes_written"] == 3
    assert result["commit_sha"] == "x"
