from __future__ import annotations

import json
import subprocess

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
    assert "smr_health_check" not in payload["missing_critical_tools"]
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


def test_codex_install_registers_managed_research_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(mcp_module.shutil, "which", lambda _name: "/usr/local/bin/codex")

    def _fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mcp_module, "_run_command", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mcp",
            "codex",
            "install",
            "--name",
            "synth_prod",
            "--hosted-url",
            "https://mcp.usesynth.ai/mcp",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        [
            "/usr/local/bin/codex",
            "mcp",
            "add",
            "synth_prod",
            "--url",
            "https://mcp.usesynth.ai/mcp",
        ]
    ]
    assert "Installed Codex MCP server 'synth_prod'." in result.output
    assert "Hosted MCP URL: https://mcp.usesynth.ai/mcp" in result.output


def test_codex_install_stdio_registers_local_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(mcp_module.shutil, "which", lambda _name: "/usr/local/bin/codex")

    def _fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mcp_module, "_run_command", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mcp",
            "codex",
            "install",
            "--transport",
            "stdio",
            "--name",
            "synth_stdio",
            "--python-executable",
            "/tmp/synth-python",
            "--backend-url",
            "https://api.usesynth.ai",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        [
            "/usr/local/bin/codex",
            "mcp",
            "add",
            "synth_stdio",
            "--env",
            "SYNTH_BACKEND_URL=https://api.usesynth.ai",
            "--",
            "/tmp/synth-python",
            "-m",
            "synth_ai.mcp.managed_research_server",
        ]
    ]
    assert "Launch command: /tmp/synth-python -m synth_ai.mcp.managed_research_server" in result.output


def test_codex_install_requires_codex_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_module.shutil, "which", lambda _name: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "codex", "install"])

    assert result.exit_code != 0
    assert "Codex CLI was not found on PATH" in result.output


def test_codex_login_invokes_codex_oauth_login(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(mcp_module.shutil, "which", lambda _name: "/usr/local/bin/codex")

    def _fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mcp_module, "_run_command", _fake_run)

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "codex", "login", "--name", "synth_prod"])

    assert result.exit_code == 0
    assert calls == [[
        "/usr/local/bin/codex",
        "mcp",
        "login",
        "synth_prod",
        "--scopes",
        "smr:read,smr:write",
    ]]
    assert "Started Codex OAuth login for 'synth_prod'." in result.output
