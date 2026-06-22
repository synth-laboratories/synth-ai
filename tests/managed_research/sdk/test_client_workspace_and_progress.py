from __future__ import annotations

import base64
from pathlib import Path

import pytest
from synth_ai.managed_research import (
    ProjectSetupAuthority,
    Provider,
    ProviderBinding,
    SmrApiError,
    SmrBranchMode,
    SmrControlClient,
    SmrCredentialProvider,
    SmrFundingSource,
    SmrLaunchPreflight,
    SmrLogicalTimeline,
    SmrRunBranchRequest,
    SmrRunPolicy,
    SmrRunPolicyAccess,
    SmrRunPolicyLimits,
    SmrToolProvider,
    UsageLimit,
    WorkspaceInputsState,
    WorkspaceUploadResult,
)
from synth_ai.managed_research.models.smr_inference_providers import SmrInferenceProvider


def test_get_workspace_download_url_calls_backend_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        return {"download_url": "https://signed", "commit_sha": "abc", "archive_key": "k1"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.get_workspace_download_url("proj_123")

    assert response["commit_sha"] == "abc"
    assert captured == {
        "method": "GET",
        "path": "/smr/projects/proj_123/workspace/download-url",
    }
    client.close()


def test_get_project_git_calls_backend_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["path"] = path
        return {"current_commit_sha": "deadbeef", "branch": "main"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.get_project_git("proj_123")

    assert response["branch"] == "main"
    assert captured["path"] == "/smr/projects/proj_123/git"
    client.close()


def test_rename_project_calls_patch_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"project_id": "proj_123", "name": "Retry transient eval failures"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.rename_project("proj_123", "  Retry transient eval failures  ")

    assert response["name"] == "Retry transient eval failures"
    assert captured == {
        "method": "PATCH",
        "path": "/smr/projects/proj_123",
        "json_body": {"name": "Retry transient eval failures"},
    }
    client.close()


def test_rename_project_rejects_blank_names(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(method: str, path: str, **kwargs):
        raise AssertionError("rename_project should reject before transport")

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    with pytest.raises(ValueError, match="project name must be non-empty"):
        client.rename_project("proj_123", "   ")
    client.close()


def test_projects_namespace_rename_delegates_to_control_client(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, str] = {}

    def fake_rename_project(project_id: str, name: str) -> dict[str, str]:
        captured["project_id"] = project_id
        captured["name"] = name
        return {"project_id": project_id, "name": name}

    monkeypatch.setattr(client, "rename_project", fake_rename_project)

    assert client.projects.rename("proj_123", "Readable project") == {
        "project_id": "proj_123",
        "name": "Readable project",
    }
    assert captured == {"project_id": "proj_123", "name": "Readable project"}
    client.close()


def test_project_lifecycle_routes_match_backend_surface(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    seen: list[tuple[str, str]] = []

    def fake_request_json(method: str, path: str, **kwargs):
        del kwargs
        seen.append((method, path))
        return {"ok": True}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.pause_project("proj_123") == {"ok": True}
    assert client.resume_project("proj_123") == {"ok": True}
    assert client.archive_project("proj_123") == {"ok": True}
    assert client.unarchive_project("proj_123") == {"ok": True}
    assert seen == [
        ("POST", "/smr/projects/proj_123/pause"),
        ("POST", "/smr/projects/proj_123/resume"),
        ("POST", "/smr/projects/proj_123/archive"),
        ("POST", "/smr/projects/proj_123/unarchive"),
    ]
    client.close()


def test_project_notes_routes_match_backend_surface(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: list[tuple[str, str, object | None]] = []

    def fake_request_json(method: str, path: str, **kwargs):
        captured.append((method, path, kwargs.get("json_body")))
        return {"notes": "ok"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.get_project_notes("proj_123") == {"notes": "ok"}
    assert client.set_project_notes("proj_123", "fresh notes") == {"notes": "ok"}
    assert client.append_project_notes("proj_123", "delta") == {"notes": "ok"}
    assert captured == [
        ("GET", "/smr/projects/proj_123/notes", None),
        ("PUT", "/smr/projects/proj_123/notes", {"notes": "fresh notes"}),
        ("POST", "/smr/projects/proj_123/notes/append", {"text": "delta"}),
    ]
    client.close()


def test_download_workspace_archive_writes_stream(monkeypatch, tmp_path: Path) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    out = tmp_path / "out.tar.gz"

    def fake_get_workspace_download_url(project_id: str) -> dict[str, str]:
        assert project_id == "proj_123"
        return {
            "download_url": "https://storage.example/archive.tgz",
            "commit_sha": "abc",
            "archive_key": "key",
        }

    monkeypatch.setattr(client, "get_workspace_download_url", fake_get_workspace_download_url)

    class _FakeStream:
        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int = 65536) -> list[bytes]:
            return [b"he", b"llo"]

    class _FakeHttpxClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _FakeHttpxClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stream(self, method: str, url: str) -> _FakeStream:
            assert method == "GET"
            assert url == "https://storage.example/archive.tgz"
            return _FakeStream()

    monkeypatch.setattr("synth_ai.managed_research.sdk.client.httpx.Client", _FakeHttpxClient)

    result = client.download_workspace_archive("proj_123", out)

    assert out.read_bytes() == b"hello"
    assert result["bytes_written"] == 5
    assert result["commit_sha"] == "abc"
    assert result["output_path"] == str(out.resolve())
    client.close()


def test_attach_source_repo_calls_workspace_input_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"ok": True}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.attach_source_repo(
        "proj_123", "https://github.com/synth/foo", default_branch="main"
    )

    assert response == {"ok": True}
    assert captured["method"] == "PUT"
    assert captured["path"] == "/smr/projects/proj_123/workspace-inputs/source-repo"
    assert captured["json_body"] == {
        "url": "https://github.com/synth/foo",
        "default_branch": "main",
    }
    client.close()


def test_upload_workspace_directory_encodes_binary_files(monkeypatch, tmp_path: Path) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")
    binary_file = tmp_path / "blob.bin"
    binary_file.write_bytes(b"\xff\x00")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"file_count": 2}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.upload_workspace_directory("proj_123", tmp_path)

    assert response == {"file_count": 2}
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/workspace-inputs/files:upload"
    payload = captured["json_body"]
    assert isinstance(payload, dict)
    files = payload["files"]
    assert files[0]["path"] == "blob.bin"
    assert files[0]["encoding"] == "base64"
    assert files[0]["content"] == base64.b64encode(b"\xff\x00").decode("ascii")
    assert files[1]["path"] == "notes.txt"
    assert files[1]["encoding"] == "utf-8"
    assert files[1]["content"] == "hello"
    client.close()


def test_workspace_inputs_namespace_returns_typed_models(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(method: str, path: str, **kwargs):
        if method == "GET":
            return {
                "project_id": "proj_123",
                "state": "ready",
                "files": [{"path": "README.md", "content_type": "text/markdown"}],
                "file_count": 1,
            }
        return {
            "project_id": "proj_123",
            "file_count": 1,
            "bytes_uploaded": 7,
            "uploaded_files": [{"path": "README.md", "encoding": "utf-8"}],
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    state = client.workspace_inputs.get("proj_123")
    uploaded = client.workspace_inputs.upload_files(
        "proj_123",
        [{"path": "README.md", "content": "updated", "content_type": "text/markdown"}],
    )

    assert isinstance(state, WorkspaceInputsState)
    assert state.file_count == 1
    assert isinstance(uploaded, WorkspaceUploadResult)
    assert uploaded.bytes_uploaded == 7
    assert uploaded.uploaded_files[0].path == "README.md"
    client.close()


def test_setup_and_launch_preflight_routes_match_remigration_surface(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    seen: list[tuple[str, str]] = []

    def fake_request_json(method: str, path: str, **kwargs):
        seen.append((method, path))
        return {"ok": True}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.get_project_setup("proj_123") == {"ok": True}
    assert client.get_launch_preflight(
        "proj_123",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=["openrouter"],
    ) == {"ok": True}
    assert seen == [
        ("GET", "/smr/projects/proj_123/setup"),
        ("POST", "/smr/projects/proj_123/launch-preflight"),
    ]
    client.close()


def test_progress_namespace_returns_typed_setup_models(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(method: str, path: str, **kwargs):
        del method
        del kwargs
        if path.endswith("/setup"):
            return {
                "state": "ready",
                "blockers": [],
                "recommended_actions": [{"tool_name": "smr_trigger_run"}],
                "workspace_inputs": {"state": "ready", "files": [], "file_count": 0},
            }
        return {
            "project_id": "proj_123",
            "clear_to_trigger": True,
            "checked": ["setup", "runtime"],
            "blockers": [],
            "providers": [{"provider": "openrouter"}],
            "capabilities": ["inference"],
            "required_capabilities": ["inference"],
            "limit": {"max_tokens": 1000},
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    setup = client.progress.get_project_setup("proj_123")
    preflight = client.progress.get_launch_preflight(
        "proj_123",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=["openrouter"],
    )

    assert isinstance(setup, ProjectSetupAuthority)
    assert setup.state == "ready"
    assert isinstance(preflight, SmrLaunchPreflight)
    assert preflight.clear_to_trigger is True
    assert preflight.providers[0].provider is Provider.OPENROUTER
    assert preflight.limit is not None
    assert preflight.limit.max_tokens == 1000
    client.close()


def test_capacity_lane_preview_calls_backend_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        return {"resolved_lane": "openai_chatgpt_pool"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.get_capacity_lane_preview("proj_123")

    assert response["resolved_lane"] == "openai_chatgpt_pool"
    assert captured == {
        "method": "GET",
        "path": "/smr/projects/proj_123/capacity-lane-preview",
    }
    client.close()


def test_run_start_blockers_uses_trigger_compatible_payload(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"clear_to_trigger": False, "blockers": [{"error_code": "smr_limit_exceeded"}]}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.get_run_start_blockers(
        "proj_123",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=[ProviderBinding(provider=Provider.OPENROUTER)],
        limit=UsageLimit(max_spend_usd=25.0),
        worker_pool_id="pool_123",
        timebox_seconds=1800,
        agent_profile="ap_worker",
        agent_model="gpt-5.4",
        agent_kind="codex",
        agent_model_params={"reasoning_effort": "high"},
        initial_runtime_messages=[
            {"body": "Check the failing CI lane.", "mode": "queue"},
        ],
        sandbox_override={"image": "synth/smr:latest"},
        run_policy=SmrRunPolicy(
            funding_source=SmrFundingSource.CUSTOMER_BYOK,
            access=SmrRunPolicyAccess(
                credential_providers=(SmrCredentialProvider.OPENROUTER,),
                inference_providers=(SmrInferenceProvider.GOOGLE,),
                tool_providers=(SmrToolProvider.TINKER,),
            ),
            limits=SmrRunPolicyLimits(total_cost_cents=2500),
        ),
        idempotency_key_run_create="idem_123",
    )

    assert response["clear_to_trigger"] is False
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/launch-preflight"
    assert captured["json_body"] == {
        "host_kind": "daytona",
        "work_mode": "directed_effort",
        "providers": [{"provider": "openrouter"}],
        "limit": {"max_spend_usd": 25.0},
        "worker_pool_id": "pool_123",
        "timebox_seconds": 1800,
        "agent_profile": "ap_worker",
        "agent_model": "gpt-5.4",
        "agent_harness": "codex",
        "agent_model_params": {"reasoning_effort": "high"},
        "initial_runtime_messages": [
            {"body": "Check the failing CI lane.", "mode": "queue"},
        ],
        "sandbox_override": {"image": "synth/smr:latest"},
        "run_policy": {
            "funding_source": "customer_byok",
            "access": {
                "credential_providers": ["openrouter"],
                "inference_providers": ["google"],
                "tool_providers": ["tinker"],
            },
            "limits": {"total_cost_cents": 2500},
        },
        "idempotency_key_run_create": "idem_123",
    }
    client.close()


def test_trigger_run_uses_project_trigger_payload(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"run_id": "run_123"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.trigger_run(
        "proj_123",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=[{"provider": "tinker", "config": {"base_model": "model-a"}}],
        worker_pool_id="pool_123",
        timebox_seconds=1800,
        agent_profile="ap_worker",
        agent_model="gpt-5.4",
        agent_kind="codex",
        agent_model_params={"reasoning_effort": "high"},
        initial_runtime_messages=[
            {"body": "Start with the highest-confidence repro.", "mode": "queue"},
        ],
        sandbox_override={"image": "synth/smr:latest"},
        run_policy={"limits": {"total_cost_cents": 1500}},
        idempotency_key="idem_legacy",
    )

    assert response["run_id"] == "run_123"
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/trigger"
    assert captured["json_body"] == {
        "host_kind": "daytona",
        "work_mode": "directed_effort",
        "providers": [{"provider": "tinker", "config": {"base_model": "model-a"}}],
        "worker_pool_id": "pool_123",
        "timebox_seconds": 1800,
        "agent_profile": "ap_worker",
        "agent_model": "gpt-5.4",
        "agent_harness": "codex",
        "agent_model_params": {"reasoning_effort": "high"},
        "initial_runtime_messages": [
            {"body": "Start with the highest-confidence repro.", "mode": "queue"},
        ],
        "sandbox_override": {"image": "synth/smr:latest"},
        "run_policy": {"limits": {"total_cost_cents": 1500}},
        "idempotency_key": "idem_legacy",
    }
    client.close()


def test_provider_key_routes_match_backend_surface(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    seen: list[tuple[str, str, object | None]] = []

    def fake_request_json(method: str, path: str, **kwargs):
        seen.append((method, path, kwargs.get("json_body")))
        return {"ok": True, "configured": method == "POST"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert (
        client.set_provider_key(
            "proj_123",
            provider=SmrCredentialProvider.OPENROUTER,
            funding_source=SmrFundingSource.CUSTOMER_BYOK,
            api_key="sk-test-key",
        )["configured"]
        is True
    )
    assert (
        client.get_provider_key_status(
            "proj_123",
            provider="openrouter",
            funding_source="customer_byok",
        )["ok"]
        is True
    )
    assert seen == [
        (
            "POST",
            "/smr/projects/proj_123/provider_keys",
            {
                "provider": "openrouter",
                "funding_source": "customer_byok",
                "api_key": "sk-test-key",
            },
        ),
        (
            "GET",
            "/smr/projects/proj_123/provider_keys/openrouter/customer_byok/status",
            None,
        ),
    ]
    client.close()


def test_list_projects_rejects_heuristic_envelopes(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(method: str, path: str, **kwargs):
        del method
        del path
        del kwargs
        return {"projects": [{"project_id": "proj_123"}]}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    with pytest.raises(SmrApiError, match="Expected list response for list_projects"):
        client.list_projects()

    client.close()


def test_run_policy_coercion_stays_typed_until_serialization(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {"run_id": "run_123"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    policy = SmrRunPolicy(
        funding_source=SmrFundingSource.CUSTOMER_BYOK,
        limits=SmrRunPolicyLimits(total_cost_cents=321),
    )

    response = client.trigger_run(
        "proj_123",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=["openrouter"],
        run_policy=policy,
    )

    assert response["run_id"] == "run_123"
    assert captured["json_body"]["run_policy"] == {
        "funding_source": "customer_byok",
        "limits": {"total_cost_cents": 321},
    }
    client.close()


def test_project_usage_routes_to_canonical_rest_surface(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        return {
            "project_id": "proj_123",
            "month_to_date": {"nominal_cents": 500},
            "last_7_days": {"nominal_cents": 125},
            "per_run": [],
            "budgets": {},
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.get_project_usage("proj_123")

    assert response.project_id == "proj_123"
    assert response.month_to_date["nominal_cents"] == 500
    assert captured["method"] == "GET"
    assert captured["path"] == "/smr/projects/proj_123/usage"
    client.close()


def test_canonical_usage_errors_raise_smr_api_error(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(method: str, path: str, **kwargs):
        del method
        del path
        del kwargs
        return {"errors": [{"message": "orgId subject must match the caller org"}]}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    with pytest.raises(SmrApiError, match="orgId subject must match the caller org"):
        client.usage.get_project_usage("proj_123")

    client.close()


def test_branch_request_rejects_invalid_checkpoint_references() -> None:
    with pytest.raises(ValueError, match="exactly one of checkpoint_id"):
        SmrRunBranchRequest()

    with pytest.raises(ValueError, match="exactly one of checkpoint_id"):
        SmrRunBranchRequest(checkpoint_id="ckpt_1", checkpoint_uri="smr://checkpoint/1")


def test_branch_request_requires_message_for_with_message() -> None:
    with pytest.raises(ValueError, match="message is required"):
        SmrRunBranchRequest(
            checkpoint_id="ckpt_1",
            mode=SmrBranchMode.WITH_MESSAGE,
        )


def test_branch_run_from_checkpoint_prefers_project_scoped_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {
            "accepted": True,
            "parent_run_id": "run_parent",
            "child_run_id": "run_child",
            "source_checkpoint_id": "ckpt_123",
            "source_checkpoint_record_id": "rec_123",
            "source_node_id": "node_123",
            "branch_message_id": "msg_123",
            "created_at": "2026-04-15T12:00:00Z",
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.branch_run_from_checkpoint(
        "run_parent",
        project_id="proj_123",
        checkpoint_id="ckpt_123",
        mode=SmrBranchMode.WITH_MESSAGE,
        message="Take a different approach.",
        reason="operator branch",
        title="Alternative path",
        source_node_id="node_123",
    )

    assert response.child_run_id == "run_child"
    assert response.branch_message_id == "msg_123"
    assert captured == {
        "method": "POST",
        "path": "/smr/projects/proj_123/runs/run_parent/branches",
        "json_body": {
            "checkpoint_id": "ckpt_123",
            "mode": "with_message",
            "message": "Take a different approach.",
            "reason": "operator branch",
            "title": "Alternative path",
            "source_node_id": "node_123",
        },
    }
    client.close()


def test_branch_run_from_checkpoint_supports_checkpoint_reference_route(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = kwargs.get("json_body")
        return {
            "accepted": True,
            "parent_run_id": "run_parent",
            "child_run_id": "run_child",
            "source_checkpoint_id": "ckpt_123",
            "created_at": "2026-04-15T12:00:00Z",
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.runs.branch_from_checkpoint(
        checkpoint_record_id="rec_123",
        mode="exact",
    )

    assert response.child_run_id == "run_child"
    assert captured == {
        "method": "POST",
        "path": "/smr/checkpoints/branches",
        "json_body": {
            "checkpoint_record_id": "rec_123",
            "mode": "exact",
        },
    }
    client.close()


def test_get_run_logical_timeline_returns_typed_model(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs):
        del kwargs
        captured["method"] = method
        captured["path"] = path
        return {
            "project_id": "proj_123",
            "run_id": "run_123",
            "generated_at": "2026-04-15T12:00:00Z",
            "run_state": "completed",
            "latest_node_id": "node_2",
            "nodes": [
                {
                    "node_id": "node_1",
                    "run_id": "run_123",
                    "created_at": "2026-04-15T11:59:00Z",
                    "logical_index": 0,
                    "kind": "checkpoint",
                    "source": "checkpoint_catalog",
                    "title": "Checkpoint saved",
                    "summary": "Saved before actor handoff.",
                    "checkpoint_id": "ckpt_123",
                    "branchable": True,
                    "detail": {"reason": "manual"},
                }
            ],
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    timeline = client.runs.get_logical_timeline("proj_123", "run_123")

    assert isinstance(timeline, SmrLogicalTimeline)
    assert timeline.nodes[0].checkpoint_id == "ckpt_123"
    assert timeline.nodes[0].detail == {"reason": "manual"}
    assert captured == {
        "method": "GET",
        "path": "/smr/projects/proj_123/runs/run_123/timeline",
    }
    client.close()
