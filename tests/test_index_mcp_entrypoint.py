"""The public Index MCP entrypoint advertises only authorized Index tools."""

from __future__ import annotations

import importlib.metadata

import pytest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.mcp.research.server import (
    INDEX_MCP_CLIENT_TIMEOUT_SECONDS,
    MCP_CLIENT_TIMEOUT_SECONDS,
    _stdio_server,
)
from synth_ai.mcp.research.tools.index import (
    INDEX_READ_TOOL_NAMES,
    INDEX_WRITE_TOOL_NAMES,
)

_AUTHENTICATED_ONLY = {
    "index_search",
    "index_search_create",
    "index_search_get",
    "index_search_result",
    "index_search_events",
    "index_search_cancel",
    "index_answer",
}


@pytest.fixture(autouse=True)
def _index_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_BACKEND_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_INDEX_MCP_WRITE_ENABLED", "false")
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)

    def refuse_transport(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("MCP tool discovery must not create a transport")

    monkeypatch.setattr(HttpTransport, "__init__", refuse_transport)


def test_installed_package_exposes_dedicated_index_command() -> None:
    scripts = {
        point.name: point.value
        for point in importlib.metadata.distribution("synth-ai").entry_points
        if point.group == "console_scripts"
    }
    assert scripts["synth-ai-index-mcp"] == "synth_ai.mcp.research.server:main_index"


def test_no_key_discovers_public_browse_only() -> None:
    names = set(_stdio_server(index_only=True).available_tool_names())
    assert names == set(INDEX_READ_TOOL_NAMES) - _AUTHENTICATED_ONLY
    assert not names.intersection(INDEX_WRITE_TOOL_NAMES)


def test_key_discovers_search_lifecycle_but_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "sk-test")
    names = set(_stdio_server(index_only=True).available_tool_names())
    assert names == set(INDEX_READ_TOOL_NAMES)


def test_writes_require_both_key_and_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_INDEX_MCP_WRITE_ENABLED", "true")
    with pytest.raises(ValueError, match="explicit SYNTH_API_KEY"):
        _stdio_server(index_only=True)

    monkeypatch.setenv("SYNTH_API_KEY", "sk-test")
    names = set(_stdio_server(index_only=True).available_tool_names())
    assert names == set(INDEX_READ_TOOL_NAMES) | set(INDEX_WRITE_TOOL_NAMES)


def test_index_entrypoint_requires_explicit_safe_backend_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYNTH_BACKEND_URL")
    with pytest.raises(ValueError, match="explicit SYNTH_BACKEND_URL"):
        _stdio_server(index_only=True)

    monkeypatch.setenv("SYNTH_BACKEND_URL", "https://user:secret@api.example.test")
    with pytest.raises(ValueError, match="without credentials"):
        _stdio_server(index_only=True)


@pytest.mark.parametrize("authenticated", [False, True])
def test_index_mcp_client_waits_for_monitored_delivery(
    monkeypatch: pytest.MonkeyPatch, authenticated: bool
) -> None:
    if authenticated:
        monkeypatch.setenv("SYNTH_API_KEY", "sk-test")
    constructed: list[dict[str, object]] = []

    def capture_transport(self: HttpTransport, **kwargs: object) -> None:
        constructed.append(kwargs)

    monkeypatch.setattr(HttpTransport, "__init__", capture_transport)
    monkeypatch.setattr(HttpTransport, "close", lambda self: None)
    server = _stdio_server(index_only=True)
    assert server._index_client_factory is not None
    with server._index_client_factory():
        pass

    assert MCP_CLIENT_TIMEOUT_SECONDS == 30.0
    assert INDEX_MCP_CLIENT_TIMEOUT_SECONDS == 120.0
    assert constructed == [
        {
            "base_url": "https://api.example.test",
            "headers": {"Authorization": "Bearer sk-test"} if authenticated else {},
            "timeout_seconds": INDEX_MCP_CLIENT_TIMEOUT_SECONDS,
        }
    ]
