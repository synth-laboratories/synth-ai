from __future__ import annotations

import io
import json

import pytest

import synth_ai.mcp.managed_research_server as managed_research_server_module
from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer, _read_message


def test_read_message_accepts_newline_delimited_jsonrpc() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "codex-mcp-client", "version": "0.110.0"},
        },
    }
    stdin = io.BytesIO(json.dumps(payload).encode("utf-8") + b"\n")

    assert _read_message(stdin) == (payload, "jsonl")


def test_read_message_accepts_content_length_frames() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {},
    }
    encoded = json.dumps(payload).encode("utf-8")
    stdin = io.BytesIO(
        f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii") + encoded
    )

    assert _read_message(stdin) == (payload, "content-length")


def test_health_check_tool_reports_backend_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str | None = None, backend_base: str | None = None) -> None:
            self.api_key = api_key
            self.backend_base = backend_base

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get_capabilities(self) -> dict[str, str]:
            return {"version": "2026.03.05", "feature_flag": "on"}

        def list_projects(self, *, limit: int = 100) -> list[dict[str, str]]:
            assert limit == 1
            return [{"id": "proj_123"}]

        def get_project_status(self, project_id: str) -> dict[str, str]:
            return {"project_id": project_id, "state": "idle"}

    monkeypatch.setattr(managed_research_server_module, "SmrControlClient", _FakeClient)
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)

    payload = ManagedResearchMcpServer()._tool_health_check(
        {"api_key": "sk_test", "project_id": "proj_123"}
    )

    assert payload["ok"] is True
    assert payload["checks"]["api_key"]["status"] == "pass"
    assert payload["checks"]["backend_ping"]["status"] == "pass"
    assert payload["checks"]["backend_ping"]["backend_version"] == "2026.03.05"
    assert payload["checks"]["project_status"]["project_id"] == "proj_123"
