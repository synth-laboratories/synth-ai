"""Unit tests for managed-research MCP server."""

from __future__ import annotations

import json

import pytest


def test_initialize_advertises_tools_capability() -> None:
    from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer

    server = ManagedResearchMcpServer()
    result = server.dispatch("initialize", {"protocolVersion": "2025-06-18"})

    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]


def test_tools_list_contains_expected_tool() -> None:
    from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer

    server = ManagedResearchMcpServer()
    result = server.dispatch("tools/list", {})

    names = {tool["name"] for tool in result["tools"]}
    assert "smr_get_project_status" in names
    assert "smr_trigger_run" in names
    assert "smr_get_starting_data_upload_urls" in names
    assert "smr_upload_starting_data" in names
    # smr_get_run_spend_entries and smr_get_run_usage_by_actor not yet implemented


def test_tools_call_unknown_tool_returns_tool_error() -> None:
    from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer

    server = ManagedResearchMcpServer()
    result = server.dispatch("tools/call", {"name": "missing_tool", "arguments": {}})

    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]


def test_tools_call_uses_sdk_client(monkeypatch) -> None:
    import synth_ai.mcp.managed_research_server as mcp_module

    class DummyClient:
        def __init__(self, api_key=None, backend_base=None):
            self.api_key = api_key
            self.backend_base = backend_base

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def get_project_status(self, project_id: str):
            return {"project_id": project_id, "state": "running"}

    monkeypatch.setattr(mcp_module, "SmrControlClient", DummyClient)

    server = mcp_module.ManagedResearchMcpServer()
    result = server.dispatch(
        "tools/call",
        {
            "name": "smr_get_project_status",
            "arguments": {
                "project_id": "project-123",
                "api_key": "sk_local",
                "backend_base": "https://api.example.com",
            },
        },
    )

    assert "isError" not in result
    payload = json.loads(result["content"][0]["text"])
    assert payload["project_id"] == "project-123"
    assert payload["state"] == "running"


def test_upload_starting_data_tool_uses_sdk_client(monkeypatch) -> None:
    import synth_ai.mcp.managed_research_server as mcp_module

    class DummyClient:
        def __init__(self, api_key=None, backend_base=None):
            self.api_key = api_key
            self.backend_base = backend_base

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def upload_starting_data_files(self, project_id: str, *, files, dataset_ref=None):
            return {
                "project_id": project_id,
                "dataset_ref": dataset_ref,
                "file_count": len(files),
            }

    monkeypatch.setattr(mcp_module, "SmrControlClient", DummyClient)

    server = mcp_module.ManagedResearchMcpServer()
    result = server.dispatch(
        "tools/call",
        {
            "name": "smr_upload_starting_data",
            "arguments": {
                "project_id": "project-123",
                "dataset_ref": "starting-data/banking77",
                "files": [
                    {
                        "path": "banking77/input_spec.json",
                        "content": "{\"kind\":\"eval_job\"}",
                        "content_type": "application/json",
                    }
                ],
            },
        },
    )

    assert "isError" not in result
    payload = json.loads(result["content"][0]["text"])
    assert payload["project_id"] == "project-123"
    assert payload["dataset_ref"] == "starting-data/banking77"
    assert payload["file_count"] == 1


@pytest.mark.skip(reason="smr_get_run_usage_by_actor not yet implemented")
def test_run_usage_by_actor_tool_uses_sdk_client(monkeypatch) -> None:
    import synth_ai.mcp.managed_research_server as mcp_module

    class DummyClient:
        def __init__(self, api_key=None, backend_base=None):
            self.api_key = api_key
            self.backend_base = backend_base

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def get_run_usage_by_actor(self, run_id: str, *, project_id=None, include_done_tasks=True):
            return {
                "run_id": run_id,
                "project_id": project_id,
                "include_done_tasks": include_done_tasks,
                "workers": [{"actor_id": "worker-a", "models": [{"provider": "openai", "model": "gpt-5.2"}]}],
                "orchestrators": [{"actor_id": "orch-1"}],
            }

    monkeypatch.setattr(mcp_module, "SmrControlClient", DummyClient)

    server = mcp_module.ManagedResearchMcpServer()
    result = server.dispatch(
        "tools/call",
        {
            "name": "smr_get_run_usage_by_actor",
            "arguments": {
                "run_id": "run-123",
                "project_id": "project-123",
                "include_done_tasks": False,
            },
        },
    )

    assert "isError" not in result
    payload = json.loads(result["content"][0]["text"])
    assert payload["run_id"] == "run-123"
    assert payload["project_id"] == "project-123"
    assert payload["include_done_tasks"] is False
    assert payload["workers"][0]["actor_id"] == "worker-a"
