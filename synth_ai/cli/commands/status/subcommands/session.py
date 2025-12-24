"""Status session subcommand."""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal

import click

from synth_ai.cli.commands.artifacts.config import resolve_backend_config
from synth_ai.cli.local.session.client import AgentSessionClient
from synth_ai.cli.local.session.exceptions import SessionNotFoundError
from synth_ai.cli.local.session.models import AgentSession


def _format_currency(value: Decimal) -> str:
    return f"${value:.2f}"


def _format_int(value: Decimal | int) -> str:
    return f"{int(value):,}"


def _session_exceeded(session: AgentSession) -> bool:
    if session.status == "limit_exceeded":
        return True
    for limit in session.limits:
        if limit.current_usage >= limit.limit_value:
            return True
    return False


def _render_session(session: AgentSession) -> None:
    click.echo(f"Session: {session.session_id}")
    click.echo(f"Status: {session.status}")
    click.echo(f"Usage: {_format_currency(session.usage.cost_usd)}")
    click.echo(f"Tokens: {_format_int(session.usage.tokens)}")
    for limit in session.limits:
        if limit.metric_type == "cost_usd":
            click.echo(f"Limit: {_format_currency(limit.limit_value)}")
        elif limit.metric_type == "tokens":
            click.echo(f"Limit: {_format_int(limit.limit_value)}")
    if _session_exceeded(session):
        click.echo("✗ EXCEEDED")
        click.echo("WARNING: Session limits exceeded!")
    else:
        click.echo("✓ OK")


@click.command("session")
@click.option("--session-id", default="", help="Session ID to inspect.")
def session_status_cmd(session_id: str) -> None:
    async def _run() -> None:
        resolved_session_id = session_id or os.getenv("SYNTH_SESSION_ID", "")
        config = resolve_backend_config()
        client = AgentSessionClient(base_url=config.base_url, api_key=config.api_key)

        if resolved_session_id:
            try:
                session = await client.get(resolved_session_id)
            except SessionNotFoundError:
                click.echo("Session not found")
                return
            _render_session(session)
            return

        sessions = await client.list(status="active")
        if not sessions:
            click.echo("No active session found")
            return
        _render_session(sessions[0])

    asyncio.run(_run())


__all__ = ["session_status_cmd"]
