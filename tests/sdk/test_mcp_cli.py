from __future__ import annotations

import json

import httpx
import pytest
from click.testing import CliRunner

import synth_ai.cli.commands.mcp as mcp_module
from synth_ai.cli.main import cli
from synth_ai.sdk.managed_research import SmrApiError


def test_run_check_rate_limit_has_explicit_hint() -> None:
    def _raise_rate_limit() -> None:
        raise SmrApiError("GET /smr/projects failed (429): Too Many Requests")

    ok, payload = mcp_module._run_check(_raise_rate_limit)

    assert ok is False
    assert payload["error_type"] == "rate_limited"
    assert payload["status_code"] == 429
    assert "Rate limited" in payload["message"]
    assert "Back off retries" in payload["hint"]


def test_run_check_timeout_has_explicit_hint() -> None:
    def _raise_timeout() -> None:
        raise httpx.ReadTimeout("request timed out")

    ok, payload = mcp_module._run_check(_raise_timeout)

    assert ok is False
    assert payload["error_type"] == "timeout"
    assert "timed out" in payload["message"]
    assert "backend reachability" in payload["hint"]


def test_mcp_tool_health_reports_server_version_and_protocol() -> None:
    payload = mcp_module._mcp_tool_health()

    assert payload["status"] in {"pass", "fail"}
    assert payload["tool_count"] > 0
    assert payload["server_name"] == "synth-ai-managed-research"
    assert isinstance(payload["server_version"], str)
    assert isinstance(payload["supported_protocol_versions"], list)
    assert payload["protocol_version"] in payload["supported_protocol_versions"]


def test_doctor_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_doctor_payload",
        lambda api_key, backend_url, project_id: {
            "ok": True,
            "backend_url": backend_url,
            "project_id": project_id,
            "checks": {
                "backend_ping": {"status": "pass"},
                "project_access": {"status": "pass"},
                "mcp_server": {"status": "pass"},
            },
            "cli": {"command": "synth-ai mcp doctor"},
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["checks"]["mcp_server"]["status"] == "pass"


def test_doctor_failure_shows_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_doctor_payload",
        lambda api_key, backend_url, project_id: {
            "ok": False,
            "backend_url": backend_url,
            "project_id": project_id,
            "checks": {
                "backend_ping": {
                    "status": "fail",
                    "message": "Rate limited by backend (HTTP 429).",
                    "hint": "Back off retries and lower polling frequency before rerunning doctor.",
                },
                "project_access": {"status": "pass"},
                "mcp_server": {"status": "pass"},
            },
            "cli": {"command": "synth-ai mcp doctor"},
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "doctor"])

    assert result.exit_code != 0
    assert "MCP doctor failed" in result.output
    assert "Rate limited by backend" in result.output
    assert "Hint: Back off retries" in result.output


def test_status_compact_outputs_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_doctor_payload",
        lambda api_key, backend_url, project_id: {
            "ok": True,
            "backend_url": backend_url,
            "project_id": project_id,
            "checks": {
                "backend_ping": {"status": "pass"},
                "project_access": {"status": "pass"},
                "mcp_server": {"status": "pass"},
            },
            "cli": {"command": "synth-ai mcp doctor"},
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "status", "--compact"])

    assert result.exit_code == 0
    assert "SMR MCP status: OK" in result.output
    assert "- backend_ping: pass" in result.output
