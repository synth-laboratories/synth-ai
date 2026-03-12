from __future__ import annotations

from typing import Any

import pytest

from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer
from synth_ai.sdk.managed_research import SmrApiError, SmrControlClient


def test_trigger_run_includes_workflow_payload() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["method"] = method
        captured["path"] = path
        captured["params"] = params
        captured["json_body"] = json_body
        captured["allow_not_found"] = allow_not_found
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    workflow = {
        "kind": "data_factory_v1",
        "profile": "founder_default",
    }

    response = client.trigger_run(
        "proj_123",
        work_mode="directed_effort",
        timebox_seconds=600,
        agent_model="gpt-4o",
        agent_kind="codex",
        workflow=workflow,
    )

    assert response == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/trigger"
    assert captured["json_body"]["workflow"] == workflow
    assert captured["json_body"]["timebox_seconds"] == 600
    assert captured["json_body"]["agent_model"] == "gpt-4o"
    assert captured["json_body"]["agent_kind"] == "codex"
    client.close()


def test_trigger_run_includes_work_mode() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]

    response = client.trigger_run(
        "proj_123",
        work_mode="directed_effort",
    )

    assert response == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/trigger"
    assert captured["json_body"]["work_mode"] == "directed_effort"
    client.close()


def test_trigger_run_requires_work_mode() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    with pytest.raises(ValueError, match="work_mode is required"):
        client.trigger_run("proj_123", work_mode=None)  # type: ignore[arg-type]

    client.close()


def test_trigger_run_rejects_unknown_work_mode() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    with pytest.raises(ValueError, match="work_mode"):
        client.trigger_run("proj_123", work_mode="freeform")

    client.close()


def test_get_run_usage_uses_run_spend_endpoint() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["method"] = method
        captured["path"] = path
        captured["params"] = params
        captured["json_body"] = json_body
        captured["allow_not_found"] = allow_not_found
        return {"run_id": "run_123", "entries": []}

    client._request_json = fake_request_json  # type: ignore[method-assign]

    response = client.get_run_usage("run_123")

    assert response == {
        "run_id": "run_123",
        "project_id": None,
        "total_cost_cents": 0,
        "total_charged_cents": 0,
        "entries": [],
    }
    assert captured["method"] == "GET"
    assert captured["path"] == "/smr/runs/run_123/spend"
    assert captured["params"] is None
    assert captured["json_body"] is None
    assert captured["allow_not_found"] is False
    client.close()


def test_get_run_usage_validates_project_membership_when_project_id_provided() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, bool]] = []

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        calls.append((method, path, allow_not_found))
        if path == "/smr/projects/proj_123/runs/run_123":
            return {"run_id": "run_123", "project_id": "proj_123"}
        if path == "/smr/runs/run_123/spend":
            return {"run_id": "run_123", "entries": []}
        raise AssertionError(f"Unexpected request path: {path}")

    client._request_json = fake_request_json  # type: ignore[method-assign]

    response = client.get_run_usage("run_123", project_id="proj_123")

    assert response == {
        "run_id": "run_123",
        "project_id": "proj_123",
        "total_cost_cents": 0,
        "total_charged_cents": 0,
        "entries": [],
    }
    assert calls == [
        ("GET", "/smr/projects/proj_123/runs/run_123", True),
        ("GET", "/smr/runs/run_123/spend", False),
    ]
    client.close()


def test_get_run_usage_normalizes_legacy_entry_only_payload() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        if path != "/smr/runs/run_123/spend":
            raise AssertionError(f"Unexpected request path: {path}")
        return {
            "run_id": "run_123",
            "entries": [
                {
                    "project_id": "proj_123",
                    "cost_cents": 12,
                    "metadata": {"billing": {"chargeable": True}},
                },
                {
                    "project_id": "proj_123",
                    "cost_cents": 30,
                    "metadata": {"billing": {"charged_amount_cents": "7"}},
                },
                {
                    "project_id": "proj_123",
                    "cost_cents": 5,
                    "metadata": {"billing": {"chargeable": False}},
                },
            ],
        }

    client._request_json = fake_request_json  # type: ignore[method-assign]

    response = client.get_run_usage("run_123")

    assert response == {
        "run_id": "run_123",
        "project_id": "proj_123",
        "total_cost_cents": 47,
        "total_charged_cents": 19,
        "entries": [
            {
                "project_id": "proj_123",
                "cost_cents": 12,
                "metadata": {"billing": {"chargeable": True}},
            },
            {
                "project_id": "proj_123",
                "cost_cents": 30,
                "metadata": {"billing": {"charged_amount_cents": "7"}},
            },
            {
                "project_id": "proj_123",
                "cost_cents": 5,
                "metadata": {"billing": {"chargeable": False}},
            },
        ],
    }
    client.close()


def test_trigger_data_factory_run_builds_expected_workflow_defaults() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        work_mode: str,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
        idempotency_key_run_create: str | None = None,
    ) -> dict[str, Any]:
        captured["project_id"] = project_id
        captured["work_mode"] = work_mode
        captured["timebox_seconds"] = timebox_seconds
        captured["workflow"] = workflow
        captured["idempotency_key_run_create"] = idempotency_key_run_create
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    response = client.trigger_data_factory_run(
        "proj_123",
        work_mode="directed_effort",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
    )

    assert response == {"run_id": "run_123"}
    assert captured["project_id"] == "proj_123"
    assert captured["work_mode"] == "directed_effort"
    assert captured["workflow"] == {
        "kind": "data_factory_v1",
        "profile": "founder_default",
        "source_mode": "synth_mcp_local",
        "targets": ["harbor"],
        "preferred_target": "harbor",
        "input": {
            "dataset_ref": "starting-data/demo",
            "bundle_manifest_path": "capture_bundle.json",
        },
        "options": {
            "strictness_mode": "warn",
        },
    }
    client.close()


class _FakeSmrClient:
    def __init__(self) -> None:
        self.trigger_run_calls: list[tuple[str, dict[str, Any]]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.upload_calls: list[tuple[str, dict[str, Any]]] = []
        self.finalize_calls: list[tuple[str, dict[str, Any]]] = []
        self.publish_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.binding_calls: list[tuple[str, str | None]] = []
        self.promote_binding_calls: list[tuple[str, dict[str, Any]]] = []
        self.pool_context_calls: list[tuple[str, str | None, str | None]] = []

    def __enter__(self) -> "_FakeSmrClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def trigger_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.trigger_run_calls.append((project_id, kwargs))
        return {"ok": True, "project_id": project_id, "kwargs": kwargs}

    def trigger_data_factory_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((project_id, kwargs))
        return {"ok": True, "project_id": project_id, "kwargs": kwargs}

    def upload_starting_data_files(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.upload_calls.append((project_id, kwargs))
        return {"ok": True}

    def data_factory_finalize(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.finalize_calls.append((project_id, kwargs))
        return {"ok": True, "finalization_job_id": "run_123"}

    def data_factory_finalize_status(self, project_id: str, job_id: str) -> dict[str, Any]:
        return {"ok": True, "project_id": project_id, "job_id": job_id}

    def data_factory_publish(self, project_id: str, job_id: str, **kwargs: Any) -> dict[str, Any]:
        self.publish_calls.append((project_id, job_id, kwargs))
        return {"ok": True, "project_id": project_id, "job_id": job_id, "kwargs": kwargs}

    def get_binding(self, project_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        self.binding_calls.append((project_id, run_id))
        return {
            "project_id": project_id,
            "binding_revision": 2,
            "pool_id": "pool_123",
            "dataset_revision": "dsrev_001",
            "runtime_kind": "sandbox_agent",
            "environment_kind": "harbor",
            "published_by_run_id": run_id or "run_123",
        }

    def promote_binding(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.promote_binding_calls.append((project_id, kwargs))
        return {
            "project_id": project_id,
            "binding_revision": 3,
            "pool_id": str(kwargs.get("pool_id") or ""),
            "dataset_revision": str(kwargs.get("dataset_revision") or ""),
            "runtime_kind": kwargs.get("runtime_kind") or "sandbox_agent",
            "environment_kind": kwargs.get("environment_kind") or "harbor",
            "published_by_run_id": kwargs.get("published_by_run_id") or "run_123",
        }

    def get_pool_context(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        self.pool_context_calls.append((project_id, run_id, task_id))
        return {
            "project_id": project_id,
            "run_id": run_id,
            "task_id": task_id,
            "active_binding": {
                "project_id": project_id,
                "binding_revision": 2,
                "pool_id": "pool_123",
                "runtime_kind": "sandbox_agent",
                "environment_kind": "harbor",
            },
            "run_pool_ledger": {
                "run_id": run_id,
                "project_id": project_id,
                "pools_created": [],
                "pool_claims": [],
                "task_assignments": [],
                "fallback_events": [],
            },
            "recommended_target": {"pool_id": "pool_123", "assignment_source": "project_binding"},
            "fallback_policy": {
                "require_pool_target": True,
                "allow_container_url_fallback": False,
                "allow_local_bootstrap": False,
            },
            "reason_code": None,
        }


def test_mcp_trigger_data_factory_rejects_empty_targets_list() -> None:
    server = ManagedResearchMcpServer()
    with pytest.raises(ValueError, match="targets"):
        server._tool_trigger_data_factory(
            {
                "project_id": "proj_123",
                "dataset_ref": "starting-data/demo",
                "bundle_manifest_path": "capture_bundle.json",
                "work_mode": "directed_effort",
                "targets": [],
            }
        )


def test_mcp_trigger_run_forwards_work_mode() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_trigger_run(
        {
            "project_id": "proj_123",
            "timebox_seconds": 300,
            "agent_kind": "codex",
            "work_mode": "open_ended_discovery",
        }
    )

    assert response["ok"] is True
    assert len(fake_client.trigger_run_calls) == 1
    project_id, kwargs = fake_client.trigger_run_calls[0]
    assert project_id == "proj_123"
    assert kwargs["timebox_seconds"] == 300
    assert kwargs["agent_kind"] == "codex"
    assert kwargs["work_mode"] == "open_ended_discovery"


def test_mcp_trigger_data_factory_calls_sdk_with_expected_defaults() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_trigger_data_factory(
        {
            "project_id": "proj_123",
            "dataset_ref": "starting-data/demo",
            "bundle_manifest_path": "capture_bundle.json",
            "work_mode": "directed_effort",
            "targets": ["harbor", "openenv"],
            "preferred_target": "harbor",
            "strictness_mode": "strict",
            "timebox_seconds": 1200,
        }
    )

    assert response["ok"] is True
    assert len(fake_client.calls) == 1
    project_id, kwargs = fake_client.calls[0]
    assert project_id == "proj_123"
    assert kwargs["dataset_ref"] == "starting-data/demo"
    assert kwargs["bundle_manifest_path"] == "capture_bundle.json"
    assert kwargs["profile"] == "founder_default"
    assert kwargs["source_mode"] == "synth_mcp_local"
    assert kwargs["work_mode"] == "directed_effort"
    assert kwargs["targets"] == ["harbor", "openenv"]
    assert kwargs["preferred_target"] == "harbor"
    assert kwargs["strictness_mode"] == "strict"
    assert kwargs["timebox_seconds"] == 1200


def test_trigger_data_factory_run_passes_runtime_env_kinds() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        work_mode: str,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
        idempotency_key_run_create: str | None = None,
    ) -> dict[str, Any]:
        captured["work_mode"] = work_mode
        captured["workflow"] = workflow
        captured["idempotency_key_run_create"] = idempotency_key_run_create
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        work_mode="directed_effort",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
        runtime_kind="sandbox_agent",
        environment_kind="harbor",
    )

    assert captured["workflow"]["runtime_kind"] == "sandbox_agent"
    assert captured["workflow"]["environment_kind"] == "harbor"
    assert captured["work_mode"] == "directed_effort"
    client.close()


def test_trigger_data_factory_run_omits_runtime_env_when_none() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        work_mode: str,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
        idempotency_key_run_create: str | None = None,
    ) -> dict[str, Any]:
        captured["work_mode"] = work_mode
        captured["workflow"] = workflow
        captured["idempotency_key_run_create"] = idempotency_key_run_create
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        work_mode="open_ended_discovery",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
    )

    assert "runtime_kind" not in captured["workflow"]
    assert "environment_kind" not in captured["workflow"]
    assert captured["work_mode"] == "open_ended_discovery"
    client.close()


def test_trigger_data_factory_run_passes_template_and_session_fields() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        work_mode: str,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
        idempotency_key_run_create: str | None = None,
    ) -> dict[str, Any]:
        captured["work_mode"] = work_mode
        captured["workflow"] = workflow
        captured["idempotency_key_run_create"] = idempotency_key_run_create
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        work_mode="directed_effort",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
        template="harbor_hardening_v1",
        source_mode="frontend_interactive",
        session_id="interactive_001",
        session_state="completed",
        session_title="Interactive task draft",
        session_notes="Initial draft complete.",
    )

    workflow = captured["workflow"]
    assert workflow["template"] == "harbor_hardening_v1"
    assert workflow["source_mode"] == "frontend_interactive"
    assert workflow["input"]["session_id"] == "interactive_001"
    assert workflow["input"]["session_state"] == "completed"
    assert workflow["input"]["session_title"] == "Interactive task draft"
    assert workflow["input"]["session_notes"] == "Initial draft complete."
    assert captured["work_mode"] == "directed_effort"
    client.close()


def test_trigger_data_factory_run_canonicalizes_legacy_aliases() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        work_mode: str,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
        idempotency_key_run_create: str | None = None,
    ) -> dict[str, Any]:
        captured["work_mode"] = work_mode
        captured["workflow"] = workflow
        captured["idempotency_key_run_create"] = idempotency_key_run_create
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        work_mode="directed_effort",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
        source_mode="mcp_local",
        targets=["harbor", "synth_container"],
        preferred_target="synth_container",
        runtime_kind="react",
        environment_kind="synth_container",
    )

    workflow = captured["workflow"]
    assert workflow["source_mode"] == "synth_mcp_local"
    assert workflow["targets"] == ["harbor", "custom_container"]
    assert workflow["preferred_target"] == "custom_container"
    assert workflow["runtime_kind"] == "react_mcp"
    assert workflow["environment_kind"] == "custom_container"
    assert captured["work_mode"] == "directed_effort"
    client.close()


def test_trigger_run_includes_idempotency_key_run_create() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["json_body"] = json_body
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    client.trigger_run(
        "proj_123",
        work_mode="directed_effort",
        idempotency_key_run_create="run-create-key",
    )
    assert captured["json_body"]["idempotency_key_run_create"] == "run-create-key"
    client.close()


def test_mcp_trigger_data_factory_forwards_idempotency_key_run_create() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    server._tool_trigger_data_factory(
        {
            "project_id": "proj_123",
            "dataset_ref": "starting-data/demo",
            "bundle_manifest_path": "capture_bundle.json",
            "work_mode": "directed_effort",
            "idempotency_key_run_create": "run-key-123",
        }
    )

    assert len(fake_client.calls) == 1
    _, kwargs = fake_client.calls[0]
    assert kwargs["work_mode"] == "directed_effort"
    assert kwargs["idempotency_key_run_create"] == "run-key-123"


def test_mcp_upload_starting_data_forwards_idempotency_key_upload() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    server._tool_upload_starting_data(
        {
            "project_id": "proj_123",
            "dataset_ref": "starting-data/demo",
            "idempotency_key_upload": "upload-key-123",
            "files": [{"path": "capture_bundle.json", "content": "{}"}],
        }
    )

    assert len(fake_client.upload_calls) == 1
    _, kwargs = fake_client.upload_calls[0]
    assert kwargs["idempotency_key_upload"] == "upload-key-123"


def test_mcp_data_factory_publish_forwards_idempotency_key_publish() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    server._tool_data_factory_publish(
        {
            "project_id": "proj_123",
            "job_id": "run_123",
            "idempotency_key_publish": "publish-key-123",
        }
    )

    assert len(fake_client.publish_calls) == 1
    _, _, kwargs = fake_client.publish_calls[0]
    assert kwargs["idempotency_key_publish"] == "publish-key-123"


def test_mcp_data_factory_finalize_calls_sdk() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    server._tool_data_factory_finalize(
        {
            "project_id": "proj_123",
            "dataset_ref": "starting-data/demo",
            "bundle_manifest_path": "capture_bundle.json",
            "idempotency_key_run_create": "run-key-123",
        }
    )

    assert len(fake_client.finalize_calls) == 1
    _, kwargs = fake_client.finalize_calls[0]
    assert kwargs["dataset_ref"] == "starting-data/demo"
    assert kwargs["idempotency_key_run_create"] == "run-key-123"


def test_mcp_data_factory_finalize_status_calls_sdk() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_data_factory_finalize_status(
        {
            "project_id": "proj_123",
            "job_id": "run_123",
        }
    )

    assert response["job_id"] == "run_123"


def test_data_factory_publish_includes_idempotency_key_publish() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["json_body"] = json_body
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    client.data_factory_publish(
        "proj_123",
        "run_123",
        reason="manual_publish",
        idempotency_key_publish="publish-key",
    )
    assert captured["json_body"]["idempotency_key_publish"] == "publish-key"
    client.close()


def test_get_starting_data_upload_urls_includes_idempotency_key_upload() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["json_body"] = json_body
        return {"uploads": []}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    client.get_starting_data_upload_urls(
        "proj_123",
        files=[{"path": "capture_bundle.json"}],
        idempotency_key_upload="upload-key",
    )
    assert captured["json_body"]["idempotency_key_upload"] == "upload-key"
    client.close()


def test_mcp_get_binding_calls_sdk() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_get_binding(
        {
            "project_id": "proj_123",
            "run_id": "run_999",
        }
    )

    assert response["project_id"] == "proj_123"
    assert response["published_by_run_id"] == "run_999"
    assert fake_client.binding_calls == [("proj_123", "run_999")]


def test_mcp_promote_binding_calls_sdk() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_promote_binding(
        {
            "project_id": "proj_123",
            "pool_id": "pool_abc",
            "dataset_revision": "dsrev_42",
            "expected_revision": 2,
            "runtime_kind": "sandbox_agent",
            "environment_kind": "harbor",
            "published_by_run_id": "run_999",
            "reason": "manual_promote",
            "idempotency_key": "idem-1",
        }
    )

    assert response["project_id"] == "proj_123"
    assert response["binding_revision"] == 3
    assert fake_client.promote_binding_calls == [
        (
            "proj_123",
            {
                "pool_id": "pool_abc",
                "dataset_revision": "dsrev_42",
                "expected_revision": 2,
                "runtime_kind": "sandbox_agent",
                "environment_kind": "harbor",
                "published_by_run_id": "run_999",
                "reason": "manual_promote",
                "idempotency_key": "idem-1",
            },
        )
    ]


def test_mcp_get_pool_context_calls_sdk() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_get_pool_context(
        {
            "project_id": "proj_123",
            "run_id": "run_999",
            "task_id": "task_abc",
        }
    )

    assert response["project_id"] == "proj_123"
    assert response["recommended_target"]["pool_id"] == "pool_123"
    assert fake_client.pool_context_calls == [("proj_123", "run_999", "task_abc")]


def test_get_binding_fallbacks_to_project_when_active_binding_not_found() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        if path.endswith("/active_binding"):
            return None
        if path.endswith("/smr/projects/proj_123"):
            return {
                "project_id": "proj_123",
                "pool_binding_revision": 3,
                "execution": {
                    "pool_id": "pool_from_execution",
                    "dataset_revision_id": "dsrev_execution",
                    "runtime_kind": "sandbox_agent",
                    "environment_kind": "harbor",
                },
                "last_published_run_id": "run_abc",
            }
        return {}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    binding = client.get_binding("proj_123")
    assert binding["binding_revision"] == 3
    assert binding["pool_id"] == "pool_from_execution"
    assert binding["dataset_revision"] == "dsrev_execution"
    assert binding["published_by_run_id"] == "run_abc"
    client.close()


def test_get_pool_context_returns_no_run_pool_entries_when_run_ledger_empty() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        if path.endswith("/active_binding"):
            return None
        if path.endswith("/smr/projects/proj_123"):
            return {
                "project_id": "proj_123",
                "execution": {},
            }
        if path.endswith("/smr/projects/proj_123/runs/run_abc"):
            return {"run_id": "run_abc", "status_detail": {}}
        return {}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    ctx = client.get_pool_context("proj_123", run_id="run_abc")
    assert ctx["recommended_target"] is None
    assert ctx["reason_code"] == "NO_RUN_POOL_ENTRIES"
    assert ctx["run_pool_ledger"]["pools_created"] == []
    assert ctx["run_pool_ledger"]["task_assignments"] == []
    client.close()


def test_get_pool_context_marks_deferred_pool_type_not_supported_yet() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        if path.endswith("/active_binding"):
            return {
                "project_id": "proj_123",
                "binding_revision": 7,
                "pool_id": "pool_hzn",
                "pool_type": "horizons_app",
                "runtime_kind": "horizons",
                "environment_kind": "harbor",
            }
        if path.endswith("/smr/projects/proj_123"):
            return {"project_id": "proj_123"}
        return {}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    ctx = client.get_pool_context("proj_123")
    assert ctx["recommended_target"] is None
    assert ctx["reason_code"] == "POOL_TYPE_NOT_SUPPORTED_YET"
    assert ctx["active_binding"]["pool_id"] == "pool_hzn"
    client.close()


def test_get_pool_context_reports_binding_api_not_available() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def fail_get_binding(project_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        raise SmrApiError("binding endpoint unavailable")

    client.get_binding = fail_get_binding  # type: ignore[method-assign]
    client.get_project = lambda project_id: {}  # type: ignore[method-assign]
    ctx = client.get_pool_context("proj_123")
    assert ctx["recommended_target"] is None
    assert ctx["reason_code"] == "BINDING_API_NOT_AVAILABLE"
    assert "binding_error" in ctx
    client.close()


def test_upload_starting_data_files_rejects_invalid_capture_bundle_schema() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    with pytest.raises(SmrApiError, match="INVALID_BUNDLE_SCHEMA"):
        client.upload_starting_data_files(
            "proj_123",
            files=[
                {
                    "path": "capture_bundle.json",
                    "content": "{}",
                    "content_type": "application/json",
                }
            ],
        )
    client.close()


def test_data_factory_api_methods_exist() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    assert hasattr(client, "data_factory_finalize")
    assert hasattr(client, "data_factory_finalize_status")
    assert hasattr(client, "data_factory_publish")
    assert hasattr(client, "get_binding")
    assert hasattr(client, "promote_binding")
    client.close()


def test_promote_binding_includes_expected_revision() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body
        return {"binding_revision": 8}

    client._request_json = fake_request_json  # type: ignore[method-assign]
    response = client.promote_binding(
        "proj_123",
        pool_id="pool_abc",
        dataset_revision="dsrev_42",
        expected_revision=7,
        runtime_kind="sandbox_agent",
        environment_kind="harbor",
        published_by_run_id="run_999",
        reason="manual_promote",
        idempotency_key="idem-1",
    )
    assert response["binding_revision"] == 8
    assert captured["method"] == "POST"
    assert captured["path"] == "/smr/projects/proj_123/active_binding/promote"
    assert captured["json_body"] == {
        "pool_id": "pool_abc",
        "dataset_revision": "dsrev_42",
        "expected_revision": 7,
        "runtime_kind": "sandbox_agent",
        "environment_kind": "harbor",
        "published_by_run_id": "run_999",
        "reason": "manual_promote",
        "idempotency_key": "idem-1",
    }
    client.close()


def test_list_run_pull_requests_maps_github_pr_artifacts() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    client.list_run_artifacts = lambda run_id, **kwargs: [  # type: ignore[method-assign]
        {
            "artifact_id": "art_1",
            "run_id": run_id,
            "project_id": "proj_123",
            "title": "PR artifact",
            "created_at": "2026-03-03T00:00:00Z",
            "uri": "https://github.com/org/repo/pull/42",
            "metadata": {
                "repo": "org/repo",
                "pr_number": 42,
                "pr_url": "https://github.com/org/repo/pull/42",
                "base_branch": "main",
                "head_branch": "smr/run-42",
                "state": "open",
            },
        }
    ]

    rows = client.list_run_pull_requests("run_123", project_id="proj_123")
    assert len(rows) == 1
    assert rows[0]["artifact_id"] == "art_1"
    assert rows[0]["repo"] == "org/repo"
    assert rows[0]["pr_number"] == 42
    assert rows[0]["pr_url"] == "https://github.com/org/repo/pull/42"
    client.close()


def test_mcp_tool_list_run_pull_requests() -> None:
    server = ManagedResearchMcpServer()

    class _FakeClient:
        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def list_run_pull_requests(
            self, run_id: str, *, project_id: str | None = None, limit: int = 100
        ) -> list[dict[str, Any]]:
            return [{"run_id": run_id, "project_id": project_id, "limit": limit}]

    server._client_from_args = lambda args: _FakeClient()  # type: ignore[method-assign]
    rows = server._tool_list_run_pull_requests(
        {"run_id": "run_123", "project_id": "proj_123", "limit": 5}
    )
    assert rows == [{"run_id": "run_123", "project_id": "proj_123", "limit": 5}]


def test_mcp_tool_get_artifact_content_utf8() -> None:
    server = ManagedResearchMcpServer()

    class _Resp:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.headers = {"content-type": "text/plain; charset=utf-8"}

    class _FakeClient:
        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get_artifact(self, artifact_id: str) -> dict[str, Any]:
            return {"artifact_id": artifact_id, "artifact_type": "report_md", "title": "Report", "uri": "s3://x"}

        def get_artifact_content_response(
            self, artifact_id: str, *, disposition: str = "inline", follow_redirects: bool = True
        ) -> _Resp:
            return _Resp(b"hello world")

    server._client_from_args = lambda args: _FakeClient()  # type: ignore[method-assign]
    payload = server._tool_get_artifact_content({"artifact_id": "art_1"})
    assert payload["encoding"] == "utf-8"
    assert payload["content"] == "hello world"
    assert payload["truncated"] is False
