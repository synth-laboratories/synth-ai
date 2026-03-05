"""MCP tooling diagnostics for Synth AI CLI."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from typing import Any

import click
import httpx
from synth_ai import __version__ as synth_ai_version
from synth_ai.cli.commands.managed_research import _connection_options, _emit
from synth_ai.mcp.managed_research_server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    ManagedResearchMcpServer,
)
from synth_ai.sdk.managed_research import SmrApiError, SmrControlClient

_CRITICAL_MCP_TOOLS = {
    "smr_health_check",
    "smr_get_project_status",
    "smr_trigger_run",
    "smr_get_run",
    "smr_list_runs",
    "smr_get_actor_status",
    "smr_search_project_logs",
}
_HTTP_STATUS_PATTERN = re.compile(r"\((\d{3})\)")
_DEFAULT_CODEX_SERVER_NAME = "synth_managed_research"
_MANAGED_RESEARCH_MCP_MODULE = "synth_ai.mcp.managed_research_server"
_DEFAULT_HOSTED_MCP_URL = "https://api.usesynth.ai/mcp"


def _extract_status_code(message: str) -> int | None:
    match = _HTTP_STATUS_PATTERN.search(message)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _format_exception(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, SmrApiError):
        message = str(exc)
        status_code = _extract_status_code(message)
        if status_code == 429:
            return {
                "error_type": "rate_limited",
                "status_code": status_code,
                "message": "Rate limited by backend (HTTP 429).",
                "hint": "Back off retries and lower polling frequency before rerunning doctor.",
            }
        if status_code in {401, 403}:
            return {
                "error_type": "auth_error",
                "status_code": status_code,
                "message": "Auth failed for backend request.",
                "hint": "Verify SYNTH_API_KEY is set, valid, and scoped for this org/project.",
            }
        if status_code is not None and status_code >= 500:
            return {
                "error_type": "backend_error",
                "status_code": status_code,
                "message": "Backend returned a server error.",
                "hint": "Retry shortly. If persistent, inspect backend health and logs.",
            }
        return {"error_type": "smr_api_error", "message": message}

    if isinstance(exc, httpx.TimeoutException):
        return {
            "error_type": "timeout",
            "message": "Request timed out while contacting backend.",
            "hint": "Confirm backend reachability and retry with stable network connectivity.",
        }

    if isinstance(exc, httpx.ConnectError):
        return {
            "error_type": "connect_error",
            "message": "Unable to connect to backend.",
            "hint": "Check SYNTH_BACKEND_URL and network routing to the Synth API.",
        }

    if isinstance(exc, httpx.NetworkError):
        return {
            "error_type": "network_error",
            "message": "Network error while contacting backend.",
            "hint": "Validate DNS/TLS/proxy settings and retry.",
        }

    return {"error_type": "unexpected_error", "message": str(exc)}


def _run_check(fn: Any, *, include_result: bool = False) -> tuple[bool, dict[str, Any]]:
    start_ms = int(time.perf_counter() * 1000)
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "fail"}
        payload.update(_format_exception(exc))
        return False, payload
    elapsed_ms = int(time.perf_counter() * 1000) - start_ms
    payload: dict[str, Any] = {"status": "pass", "elapsed_ms": elapsed_ms}
    if include_result:
        payload["result"] = result
    return True, payload


def _mcp_tool_health() -> dict[str, Any]:
    server = ManagedResearchMcpServer()
    tools = server.available_tool_names()
    missing = sorted(set(_CRITICAL_MCP_TOOLS) - set(tools))
    return {
        "status": "pass" if not missing else "fail",
        "tool_count": len(tools),
        "missing_critical_tools": missing,
        "server_name": "synth-ai-managed-research",
        "server_version": synth_ai_version,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "supported_protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
    }


def _annotate_backend_ping(check: dict[str, Any]) -> dict[str, Any]:
    result = check.get("result")
    if isinstance(result, dict):
        for key in ("version", "backend_version", "api_version", "build_sha"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                check["backend_version"] = value.strip()
                break
        check["capability_keys"] = sorted(result.keys())
    check.pop("result", None)
    return check


def _doctor_payload(
    api_key: str | None,
    backend_url: str,
    project_id: str | None,
) -> dict[str, Any]:
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        backend_ping_ok, backend_ping = _run_check(
            lambda: client.get_capabilities(),
            include_result=True,
        )
        backend_ping = _annotate_backend_ping(backend_ping)
        projects_ok, projects = _run_check(
            lambda: client.list_projects(limit=1),
        )
        tool_data = _mcp_tool_health()
        checks = {
            "backend_ping": backend_ping,
            "project_access": projects,
            "mcp_server": tool_data,
        }

        if project_id:
            project_ok, project_status = _run_check(
                lambda: client.get_project_status(project_id),
            )
            checks["project_status"] = project_status
            required_pass = (
                backend_ping_ok and projects_ok and project_ok and tool_data["status"] == "pass"
            )
        else:
            required_pass = backend_ping_ok and projects_ok and tool_data["status"] == "pass"

        return {
            "ok": required_pass,
            "backend_url": backend_url,
            "project_id": project_id,
            "checks": checks,
            "cli": {"command": "synth-ai mcp doctor"},
            "tooling": {
                "cli_version": synth_ai_version,
                "mcp_server": tool_data["server_name"],
                "mcp_server_version": tool_data["server_version"],
            },
        }


def _print_human_summary(payload: dict[str, Any], *, compact: bool) -> None:
    checks = payload.get("checks", {})
    click.echo(f"SMR MCP status: {'OK' if payload.get('ok') else 'ISSUES DETECTED'}")
    for name, check in checks.items():
        status = check.get("status", "unknown")
        if status == "pass":
            check_text = "pass"
        elif status == "fail":
            check_text = "fail"
        else:
            check_text = str(status)
        click.echo(
            f"- {name}: {check_text}"
            + (f" ({check.get('message')})" if check.get("message") else "")
        )
        if compact:
            continue
        if check.get("hint"):
            click.echo(f"  hint: {check['hint']}")
        if name == "backend_ping" and check.get("backend_version"):
            click.echo(f"  backend_version: {check['backend_version']}")
        if name == "mcp_server" and check.get("missing_critical_tools"):
            click.echo(f"  missing: {', '.join(check['missing_critical_tools'])}")
        if name == "mcp_server":
            click.echo(
                "  mcp_server: "
                f"{check.get('server_name')}@{check.get('server_version')} "
                f"(protocol {check.get('protocol_version')})"
            )


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _format_command_failure(result: subprocess.CompletedProcess[str]) -> str:
    output = (result.stderr or result.stdout or "").strip()
    return output or f"command exited with status {result.returncode}"


@click.group(name="mcp")
def mcp() -> None:
    """SMR MCP diagnostics and connectivity checks."""


@mcp.command(name="doctor")
@_connection_options
@click.option("--project-id", help="Optional project id to validate project status endpoint.")
def doctor(
    api_key: str | None, backend_url: str, json_output: bool, project_id: str | None
) -> None:
    """Run MCP health checks and fail on any issue."""
    try:
        payload = _doctor_payload(
            api_key=api_key,
            backend_url=backend_url,
            project_id=project_id,
        )
    except click.ClickException:
        raise
    except Exception as exc:  # noqa: BLE001
        if json_output:
            _emit({"ok": False, "error": str(exc)}, json_output=True)
            return
        raise click.ClickException(str(exc)) from None
    if not payload["ok"]:
        if json_output:
            _emit(payload, json_output=True)
            return
        failed: list[str] = []
        for name, details in payload.get("checks", {}).items():
            if details.get("status") == "pass":
                continue
            message = details.get("message") or "check failed"
            hint = details.get("hint")
            detail_text = f"{name}: {message}"
            if hint:
                detail_text += f" Hint: {hint}"
            failed.append(detail_text)
        raise click.ClickException("MCP doctor failed:\n- " + "\n- ".join(failed))
    _emit(payload, json_output=json_output)


@mcp.command(name="status")
@_connection_options
@click.option("--project-id", help="Optional project id to include project status check.")
@click.option(
    "--compact",
    is_flag=True,
    default=False,
    help="Print compact one-line output.",
)
def status(
    api_key: str | None,
    backend_url: str,
    json_output: bool,
    project_id: str | None,
    compact: bool,
) -> None:
    """Show a quick MCP + backend status snapshot."""
    try:
        payload = _doctor_payload(api_key=api_key, backend_url=backend_url, project_id=project_id)
    except Exception as exc:  # noqa: BLE001
        if json_output:
            _emit({"ok": False, "error": str(exc)}, json_output=True)
            return
        raise click.ClickException(str(exc)) from None

    if json_output:
        _emit(payload, json_output=True)
        return

    _print_human_summary(payload, compact=compact)


@mcp.group(name="codex")
def codex() -> None:
    """Install and manage the Synth managed-research MCP server in Codex."""


@codex.command(name="install")
@click.option(
    "--name",
    default=_DEFAULT_CODEX_SERVER_NAME,
    show_default=True,
    help="Codex MCP server name to register.",
)
@click.option(
    "--transport",
    type=click.Choice(["hosted", "stdio"]),
    default="hosted",
    show_default=True,
    help="Install the hosted MCP endpoint or the local stdio server.",
)
@click.option(
    "--hosted-url",
    default=_DEFAULT_HOSTED_MCP_URL,
    show_default=True,
    help="Hosted MCP URL to register when using --transport hosted.",
)
@click.option(
    "--codex-binary",
    default="codex",
    show_default=True,
    help="Codex CLI binary to invoke.",
)
@click.option(
    "--python-executable",
    default=sys.executable,
    show_default=True,
    help="Python interpreter that has synth-ai installed.",
)
@click.option(
    "--backend-url",
    help="Optional backend URL to persist into the Codex MCP config.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace any existing Codex MCP config with the same name.",
)
def codex_install(
    name: str,
    transport: str,
    hosted_url: str,
    codex_binary: str,
    python_executable: str,
    backend_url: str | None,
    force: bool,
) -> None:
    """Register the managed-research MCP server with Codex CLI."""
    codex_path = shutil.which(codex_binary)
    if codex_path is None:
        raise click.ClickException(
            "Codex CLI was not found on PATH. Install Codex, then rerun `synth-ai mcp codex install`."
        )

    if force:
        _run_command([codex_path, "mcp", "remove", name])

    command = [codex_path, "mcp", "add", name]
    if transport == "hosted":
        command.extend(["--url", hosted_url.strip()])
    else:
        if backend_url:
            command.extend(["--env", f"SYNTH_BACKEND_URL={backend_url.strip()}"])
        command.extend(
            [
                "--",
                python_executable,
                "-m",
                _MANAGED_RESEARCH_MCP_MODULE,
            ]
        )

    result = _run_command(command)
    if result.returncode != 0:
        detail = _format_command_failure(result)
        if not force and "already exists" in detail.lower():
            detail += " Re-run with --force to replace the existing MCP config."
        raise click.ClickException(f"Failed to install Codex MCP server '{name}': {detail}")

    click.echo(f"Installed Codex MCP server '{name}'.")
    if transport == "hosted":
        click.echo(f"Hosted MCP URL: {hosted_url.strip()}")
        click.echo("Next steps:")
        click.echo("1. Finish the Codex OAuth flow in the browser if prompted.")
        click.echo("2. Run `codex mcp get " + name + "` to inspect the saved config.")
        click.echo("3. Start Codex and call `smr_projects_list` or `smr_project_trigger_run`.")
    else:
        click.echo(f"Launch command: {python_executable} -m {_MANAGED_RESEARCH_MCP_MODULE}")
        click.echo("Next steps:")
        click.echo("1. Run `synth-ai setup` or export SYNTH_API_KEY before launching Codex.")
        click.echo("2. Run `codex mcp get " + name + "` to inspect the saved config.")
        click.echo("3. Start Codex and call `smr_health_check` or `smr_list_projects`.")


@codex.command(name="login")
@click.option(
    "--name",
    default=_DEFAULT_CODEX_SERVER_NAME,
    show_default=True,
    help="Codex MCP server name to authenticate.",
)
@click.option(
    "--codex-binary",
    default="codex",
    show_default=True,
    help="Codex CLI binary to invoke.",
)
@click.option(
    "--scopes",
    default="smr:read,smr:write",
    show_default=True,
    help="Comma-separated OAuth scopes to request.",
)
def codex_login(name: str, codex_binary: str, scopes: str) -> None:
    """Trigger Codex OAuth login for the hosted managed-research MCP server."""
    codex_path = shutil.which(codex_binary)
    if codex_path is None:
        raise click.ClickException(
            "Codex CLI was not found on PATH. Install Codex, then rerun `synth-ai mcp codex login`."
        )

    command = [codex_path, "mcp", "login", name]
    normalized_scopes = ",".join(entry.strip() for entry in scopes.split(",") if entry.strip())
    if normalized_scopes:
        command.extend(["--scopes", normalized_scopes])
    result = _run_command(command)
    if result.returncode != 0:
        raise click.ClickException(
            f"Failed to start Codex OAuth login for '{name}': {_format_command_failure(result)}"
        )
    click.echo(f"Started Codex OAuth login for '{name}'.")
