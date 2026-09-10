"""``research_confirm_workspace_push`` gives MCP-driven ingress a done signal.

The SDK has ``client.projects.workspace.confirm_push`` returning the typed
``WorkspacePushConfirmationReceipt``, but MCP previously lacked it, so coding
agents driving workspace ingress over MCP had no confirmation receipt. The
tool is write-scoped and wraps the SDK call one-to-one. Confirming a push on
a project bound to an open Sync session auto-records a kit-association
receipt server-side (backend PR #1148); the tool description says so.
"""

from __future__ import annotations

from typing import Any

import pytest
from synth_ai.mcp.research.registry import (
    _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME,
    WRITE_SCOPES,
    build_tool_registry,
)
from synth_ai.mcp.research.tools.workspace_inputs import build_workspace_input_tools

TOOL_NAME = "research_confirm_workspace_push"
_COMMIT_SHA = "a" * 40


class _Receipt:
    def __init__(self) -> None:
        self.payload = {
            "schema_version": "synth.workspace-push-confirmation-receipt.v1",
            "receipt_id": "rcpt_1",
        }

    def to_wire(self) -> dict[str, Any]:
        return dict(self.payload)


class _RecordingWorkspaceAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def confirm_push(self, project_id: Any, **kwargs: Any) -> _Receipt:
        self.calls.append((project_id, kwargs))
        return _Receipt()


class _StubClient:
    def __init__(self) -> None:
        self.projects = type("_Projects", (), {})()
        self.projects.workspace = _RecordingWorkspaceAPI()

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


class _StubClientFactory:
    def __init__(self) -> None:
        self.client = _StubClient()
        self.invocations = 0

    def __call__(self, args: dict[str, Any]) -> _StubClient:
        self.invocations += 1
        return self.client


@pytest.fixture()
def factory() -> _StubClientFactory:
    return _StubClientFactory()


def _tool(factory: _StubClientFactory) -> Any:
    tools = {tool.name: tool for tool in build_workspace_input_tools(factory)}
    return tools[TOOL_NAME]


def test_registered_with_write_scopes(factory: _StubClientFactory) -> None:
    tool = _tool(factory)
    assert tool.required_scopes == WRITE_SCOPES
    assert _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME[TOOL_NAME] == WRITE_SCOPES
    registry = build_tool_registry(build_workspace_input_tools(factory))
    assert registry[TOOL_NAME].required_scopes == WRITE_SCOPES


def test_advertised_by_stable_stdio_server() -> None:
    from synth_ai.mcp.research.server import ResearchMcpServer

    assert TOOL_NAME in ResearchMcpServer(api_key="sk-test").available_tool_names()


def test_input_schema_requires_full_identity(factory: _StubClientFactory) -> None:
    schema = _tool(factory).input_schema
    assert set(schema["required"]) == {"project_id", "commit_sha", "archive_key", "run_id"}
    assert schema["additionalProperties"] is False


def test_description_names_backend_owned_workspace_confirmation_receipt(
    factory: _StubClientFactory,
) -> None:
    description = _tool(factory).description
    assert "WorkspacePushConfirmationReceipt" in description
    assert "project authority" in description
    assert "already-pushed" in description


def test_handler_requires_every_argument(factory: _StubClientFactory) -> None:
    handler = _tool(factory).handler
    complete = {
        "project_id": "proj_1",
        "commit_sha": _COMMIT_SHA,
        "archive_key": "archives/proj_1/abc.tar.zst",
        "run_id": "run_1",
    }
    for missing in complete:
        args = {key: value for key, value in complete.items() if key != missing}
        with pytest.raises(ValueError, match=f"'{missing}'"):
            handler(args)
    assert factory.invocations == 0


def test_handler_wires_confirm_push_to_sdk(factory: _StubClientFactory) -> None:
    handler = _tool(factory).handler
    result = handler(
        {
            "project_id": "proj_1",
            "commit_sha": _COMMIT_SHA,
            "archive_key": "archives/proj_1/abc.tar.zst",
            "run_id": "run_1",
        }
    )
    assert result["receipt_id"] == "rcpt_1"
    ((project_id, kwargs),) = factory.client.projects.workspace.calls
    assert str(project_id) == "proj_1"
    assert kwargs == {
        "commit_sha": _COMMIT_SHA,
        "archive_key": "archives/proj_1/abc.tar.zst",
        "run_id": "run_1",
    }
