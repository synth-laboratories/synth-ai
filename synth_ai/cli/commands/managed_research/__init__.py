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
def project_status(
    project_id: str, api_key: str | None, backend_url: str, json_output: bool
) -> None:
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
@click.option(
    "--work-mode",
    type=click.Choice(["open_ended_discovery", "directed_effort"], case_sensitive=False),
    required=True,
    help="Required run work mode.",
)
@_connection_options
def trigger(
    project_id: str,
    timebox_seconds: int | None,
    work_mode: str,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Trigger a run for a managed research project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.trigger_run(
            project_id,
            work_mode=work_mode,
            timebox_seconds=timebox_seconds,
        )
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


@managed_research.command(name="jobs")
@click.option("--project-id", help="Optional project_id filter.")
@click.option("--state", help="Optional run-state filter (single or comma-separated).")
@click.option("--active", is_flag=True, help="Only include active runs.")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows to return.")
@_connection_options
def jobs(
    project_id: str | None,
    state: str | None,
    active: bool,
    limit: int,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """List org-level jobs feed (SMR runs), optionally filtered by project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.list_jobs(
            project_id=project_id,
            state=state,
            active_only=active,
            limit=limit,
        )
        _emit(payload, json_output=json_output)


@managed_research.command(name="actor-status")
@click.argument("project_id")
@click.option("--run-id", help="Optional run id filter.")
@_connection_options
def actor_status(
    project_id: str,
    run_id: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Get unified actor status for orchestrator and workers."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.get_actor_status(project_id, run_id=run_id)
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


@managed_research.command(name="actor-action")
@click.argument("action", type=click.Choice(["pause", "resume"]))
@click.argument("project_id")
@click.argument("run_id")
@click.argument("actor_id")
@click.option("--reason", help="Optional reason to include in the control request.")
@click.option("--idempotency-key", help="Optional idempotency key.")
@_connection_options
def actor_action(
    action: str,
    project_id: str,
    run_id: str,
    actor_id: str,
    reason: str | None,
    idempotency_key: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Pause or resume a specific worker/orchestrator actor."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        payload = client.control_actor(
            project_id,
            run_id,
            actor_id,
            action=action,  # type: ignore[arg-type]
            reason=reason,
            idempotency_key=idempotency_key,
        )
        _emit(payload, json_output=json_output)


@managed_research.command(name="respond-question")
@click.argument("run_id")
@click.argument("question_id")
@click.option("--response-text", required=True, help="Response text to submit.")
@click.option("--project-id", help="Optional project_id for project-scoped strict route.")
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
@click.option("--project-id", help="Optional project_id for project-scoped strict route.")
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
    import urllib.parse
    import webbrowser

    captured: dict[str, str] = {}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = urllib.parse.urlparse(self.path).query
            params = dict(urllib.parse.parse_qsl(qs))
            captured.update(params)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>GitHub connected. You can close this tab.</h2></body></html>"
            )

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

        click.echo("Opening browser for GitHub authorization...")
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


# -- Linear project-level integration commands ------------------------------


@managed_research.group(name="linear")
def linear_group() -> None:
    """Manage project-level Linear connection."""


@linear_group.command(name="status")
@click.argument("project_id")
@_connection_options
def linear_status(
    project_id: str, api_key: str | None, backend_url: str, json_output: bool
) -> None:
    """Show project-level Linear connection status."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.linear_status(project_id), json_output=json_output)


@linear_group.command(name="connect")
@click.argument("project_id")
@click.option("--port", type=int, default=9877, help="Local port for OAuth callback.")
@_connection_options
def linear_connect(
    project_id: str,
    port: int,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Connect Linear via OAuth (opens browser)."""
    import http.server
    import urllib.parse
    import webbrowser

    captured: dict[str, str] = {}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = urllib.parse.urlparse(self.path).query
            params = dict(urllib.parse.parse_qsl(qs))
            captured.update(params)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Linear connected. You can close this tab.</h2></body></html>"
            )

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass  # suppress request logs

    redirect_uri = f"http://localhost:{port}/callback"
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        start_resp = client.linear_oauth_start(project_id=project_id, redirect_uri=redirect_uri)
        authorize_url = start_resp.get("authorize_url")
        if not authorize_url:
            raise click.ClickException("Backend did not return authorize_url")

        httpd = http.server.HTTPServer(("127.0.0.1", port), CallbackHandler)
        httpd.timeout = 120

        click.echo("Opening browser for Linear authorization...")
        click.echo(f"  (listening on http://localhost:{port}/callback)")
        webbrowser.open(authorize_url)

        # Wait for single callback request.
        httpd.handle_request()
        httpd.server_close()

        code = captured.get("code", "")
        state = captured.get("state", "")
        if not code:
            raise click.ClickException("No authorization code received from Linear.")

        result = client.linear_oauth_callback(
            project_id=project_id,
            code=code,
            state=state,
            redirect_uri=redirect_uri,
        )
        _emit(result, json_output=json_output)
        if not json_output:
            click.echo("Linear connected.")


@linear_group.command(name="disconnect")
@click.argument("project_id")
@_connection_options
def linear_disconnect(
    project_id: str, api_key: str | None, backend_url: str, json_output: bool
) -> None:
    """Disconnect project-level Linear credential."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.linear_disconnect(project_id), json_output=json_output)


@linear_group.command(name="teams")
@click.argument("project_id")
@_connection_options
def linear_teams(project_id: str, api_key: str | None, backend_url: str, json_output: bool) -> None:
    """List available Linear teams for a connected project."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.linear_list_teams(project_id), json_output=json_output)


@managed_research.group(name="codex")
def codex_group() -> None:
    """Manage global Codex subscription connection."""


@codex_group.command(name="status")
@click.option("--project-id", help="Optional project id to read project-bound state.")
@_connection_options
def codex_status(
    project_id: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Show Codex subscription connection status."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.chatgpt_connection_status(project_id=project_id), json_output=json_output)


@codex_group.command(name="connect-start")
@click.option("--sandbox-agent-url", help="Optional connector URL override.")
@click.option("--provider-id", help="Optional connector provider id override.")
@click.option("--external-account-hint", help="Optional account hint to store.")
@_connection_options
def codex_connect_start(
    sandbox_agent_url: str | None,
    provider_id: str | None,
    external_account_hint: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Start Codex subscription login flow."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(
            client.chatgpt_connect_start(
                sandbox_agent_url=sandbox_agent_url,
                provider_id=provider_id,
                external_account_hint=external_account_hint,
            ),
            json_output=json_output,
        )


@codex_group.command(name="connect-complete")
@click.option("--code", help="Optional OAuth code for code-based flows.")
@click.option("--sandbox-agent-url", help="Optional connector URL override.")
@_connection_options
def codex_connect_complete(
    code: str | None,
    sandbox_agent_url: str | None,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Complete Codex subscription login flow."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(
            client.chatgpt_connect_complete(code=code, sandbox_agent_url=sandbox_agent_url),
            json_output=json_output,
        )


@codex_group.command(name="disconnect")
@_connection_options
def codex_disconnect(api_key: str | None, backend_url: str, json_output: bool) -> None:
    """Disconnect global Codex subscription."""
    with SmrControlClient(api_key=api_key, backend_base=backend_url) as client:
        _emit(client.chatgpt_disconnect(), json_output=json_output)


__all__ = ["managed_research"]
