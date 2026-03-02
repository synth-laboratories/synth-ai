from __future__ import annotations

from typing import Any

import pytest

from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer
from synth_ai.sdk.managed_research import SmrControlClient


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


def test_trigger_data_factory_run_builds_expected_workflow_defaults() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured["project_id"] = project_id
        captured["timebox_seconds"] = timebox_seconds
        captured["workflow"] = workflow
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    response = client.trigger_data_factory_run(
        "proj_123",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
    )

    assert response == {"run_id": "run_123"}
    assert captured["project_id"] == "proj_123"
    assert captured["workflow"] == {
        "kind": "data_factory_v1",
        "profile": "founder_default",
        "source_mode": "mcp_local",
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
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "_FakeSmrClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def trigger_data_factory_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((project_id, kwargs))
        return {"ok": True, "project_id": project_id, "kwargs": kwargs}


def test_mcp_trigger_data_factory_rejects_empty_targets_list() -> None:
    server = ManagedResearchMcpServer()
    with pytest.raises(ValueError, match="targets"):
        server._tool_trigger_data_factory(
            {
                "project_id": "proj_123",
                "dataset_ref": "starting-data/demo",
                "bundle_manifest_path": "capture_bundle.json",
                "targets": [],
            }
        )


def test_mcp_trigger_data_factory_calls_sdk_with_expected_defaults() -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeSmrClient()
    server._client_from_args = lambda args: fake_client  # type: ignore[method-assign]

    response = server._tool_trigger_data_factory(
        {
            "project_id": "proj_123",
            "dataset_ref": "starting-data/demo",
            "bundle_manifest_path": "capture_bundle.json",
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
    assert kwargs["source_mode"] == "mcp_local"
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
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured["workflow"] = workflow
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
        runtime_kind="sandbox_agent",
        environment_kind="harbor",
    )

    assert captured["workflow"]["runtime_kind"] == "sandbox_agent"
    assert captured["workflow"]["environment_kind"] == "harbor"
    client.close()


def test_trigger_data_factory_run_omits_runtime_env_when_none() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_trigger_run(
        project_id: str,
        *,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured["workflow"] = workflow
        return {"run_id": "run_123"}

    client.trigger_run = fake_trigger_run  # type: ignore[method-assign]

    client.trigger_data_factory_run(
        "proj_123",
        dataset_ref="starting-data/demo",
        bundle_manifest_path="capture_bundle.json",
    )

    assert "runtime_kind" not in captured["workflow"]
    assert "environment_kind" not in captured["workflow"]
    client.close()


def test_data_factory_api_methods_exist() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    assert hasattr(client, "data_factory_finalize")
    assert hasattr(client, "data_factory_finalize_status")
    assert hasattr(client, "data_factory_publish")
    client.close()
