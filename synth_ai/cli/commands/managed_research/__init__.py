"""Managed Research CLI commands."""

from __future__ import annotations

import json
from typing import Any, Callable

import click
from synth_ai.core.utils.urls import resolve_synth_backend_url
from synth_ai.sdk.managed_research import SmrControlClient


def _emit(payload: Any, *, json_output: bool) -> None:
    if json_output:
        click.echo(json.dumps(payload, separators=(",", ":"), default=str))
    else:
        click.echo(json.dumps(payload, indent=2, default=str))


def _connection_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn = click.option(
        "--api-key",
        envvar="SYNTH_API_KEY",
        help="Synth API key.",
    )(fn)
    fn = click.option(
        "--backend-url",
        envvar="SYNTH_BACKEND_URL",
        default=resolve_synth_backend_url(),
        show_default=True,
        help="Synth backend URL.",
    )(fn)
    fn = click.option(
        "--json",
        "json_output",
        is_flag=True,
        help="Emit compact JSON output.",
    )(fn)
    return fn


@click.group(name="managed-research")
def managed_research() -> None:
    """Control Synth Managed Research projects and runs."""


@managed_research.command(name="project-status")
@click.argument("project_id")
@_connection_options
def project_status(project_id: str, api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Get current status for a managed research project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.get_project_status(project_id), json_output=json_output)


@managed_research.command(name="trigger")
@click.argument("project_id")
@click.option(
    "--timebox-seconds",
    type=int,
    help="Optional run timebox in seconds.",
)
@_connection_options
def trigger(
    project_id: str,
    timebox_seconds: int | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Trigger a run for a managed research project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.trigger_run(project_id, timebox_seconds=timebox_seconds)
        _emit(payload, json_output=json_output)


@managed_research.command(name="runs")
@click.argument("project_id")
@click.option(
    "--active",
    is_flag=True,
    help="Only include active runs.",
)
@_connection_options
def runs(
    project_id: str,
    active: bool,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """List runs for a managed research project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.list_active_runs(project_id) if active else client.list_runs(project_id)
        _emit(payload, json_output=json_output)


@managed_research.command(name="run-action")
@click.argument("action", type=click.Choice(["pause", "resume", "stop"]))
@click.argument("run_id")
@_connection_options
def run_action(
    action: str,
    run_id: str,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Pause, resume, or stop a run."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        if action == "pause":
            payload = client.pause_run(run_id)
        elif action == "resume":
            payload = client.resume_run(run_id)
        else:
            payload = client.stop_run(run_id)
        _emit(payload, json_output=json_output)


@managed_research.command(name="respond-question")
@click.argument("run_id")
@click.argument("question_id")
@click.option("--response-text", required=True, help="Response text to submit.")
@click.option("--project-id", help="Optional project_id for project-scoped fallback route.")
@_connection_options
def respond_question(
    run_id: str,
    question_id: str,
    response_text: str,
    project_id: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Respond to a run question."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.respond_question(
            run_id,
            question_id,
            response_text=response_text,
            project_id=project_id,
        )
        _emit(payload, json_output=json_output)


@managed_research.command(name="approval")
@click.argument("decision", type=click.Choice(["approve", "deny"]))
@click.argument("run_id")
@click.argument("approval_id")
@click.option("--comment", help="Optional decision comment.")
@click.option("--project-id", help="Optional project_id for project-scoped fallback route.")
@_connection_options
def approval(
    decision: str,
    run_id: str,
    approval_id: str,
    comment: str | None,
    project_id: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Approve or deny a run approval request."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        if decision == "approve":
            payload = client.approve(run_id, approval_id, comment=comment, project_id=project_id)
        else:
            payload = client.deny(run_id, approval_id, comment=comment, project_id=project_id)
        _emit(payload, json_output=json_output)


@managed_research.command(name="mcp-server")
def mcp_server() -> None:
    """Run the Managed Research MCP server over stdio."""
    from synth_ai.mcp.managed_research_server import main

    main()


# -- GitHub org-level integration commands ---------------------------------


@managed_research.group(name="github")
def github_group() -> None:
    """Manage org-level GitHub connection."""


@github_group.command(name="status")
@_connection_options
def github_status(api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Show org-level GitHub connection status."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.github_org_status(), json_output=json_output)


@github_group.command(name="connect")
@click.option("--port", type=int, default=9876, help="Local port for OAuth callback.")
@_connection_options
def github_connect(port: int, api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Connect GitHub via OAuth (opens browser)."""
    import http.server
    import threading
    import urllib.parse
    import webbrowser

    captured: dict[str, str] = {}
    server_ready = threading.Event()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = urllib.parse.urlparse(self.path).query
            params = dict(urllib.parse.parse_qsl(qs))
            captured.update(params)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>GitHub connected. You can close this tab.</h2></body></html>")

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass  # suppress request logs

    redirect_uri = f"http://localhost:{port}/callback"
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        start_resp = client.github_org_oauth_start(redirect_uri=redirect_uri)
        authorize_url = start_resp.get("authorize_url")
        if not authorize_url:
            raise click.ClickException("Backend did not return authorize_url")

        httpd = http.server.HTTPServer(("127.0.0.1", port), CallbackHandler)
        httpd.timeout = 120

        click.echo(f"Opening browser for GitHub authorization...")
        click.echo(f"  (listening on http://localhost:{port}/callback)")
        webbrowser.open(authorize_url)

        # Wait for single callback request.
        httpd.handle_request()
        httpd.server_close()

        code = captured.get("code", "")
        state = captured.get("state", "")
        if not code:
            raise click.ClickException("No authorization code received from GitHub.")

        result = client.github_org_oauth_callback(code=code, state=state, redirect_uri=redirect_uri)
        _emit(result, json_output=json_output)
        gh_user = result.get("github_user")
        if not json_output:
            click.echo(f"Connected as @{gh_user}" if gh_user else "Connected.")


@github_group.command(name="disconnect")
@_connection_options
def github_disconnect(api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Disconnect org-level GitHub credential."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.github_org_disconnect(), json_output=json_output)


@github_group.command(name="link")
@click.argument("project_id")
@_connection_options
def github_link(project_id: str, api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Link a project to the org-level GitHub credential."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.github_link_org(project_id), json_output=json_output)


__all__ = ["managed_research"]
