"""Research hero smoke commands."""

from __future__ import annotations

import json
import os

import click

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base


def _resolve_backend_url(backend_url: str | None) -> str:
    return normalize_backend_base(
        backend_url or os.environ.get("SYNTH_BACKEND_URL") or BACKEND_URL_BASE
    )


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or get_api_key(required=False) or "").strip()
    if not resolved:
        raise click.ClickException("api_key is required (pass --api-key or set SYNTH_API_KEY)")
    return resolved


@click.group()
def research() -> None:
    """Managed Research hero SDK smoke commands."""


@research.group()
def limits() -> None:
    """Org limits readout."""


@limits.command("get")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def limits_get(api_key: str | None, backend_url: str | None) -> None:
    """Fetch org limits via ``client.research.limits.get()``."""
    from synth_ai import SynthClient

    client = SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )
    payload = client.research.limits.get()
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@research.group()
def tag() -> None:
    """Factory Tag smoke loop."""


@tag.command("smoke")
@click.option(
    "--request", default="Summarize factory status", show_default=True, help="Tag session request."
)
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def tag_smoke(request: str, api_key: str | None, backend_url: str | None) -> None:
    """Create a Tag session, send a message, and print scope metadata."""
    from synth_ai import SynthClient

    client = SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )
    tag_api = client.research.factories.tag
    session = tag_api.sessions.create(request)
    tag_api.sessions.messages.send(session.session_id, "Smoke check-in")
    scope = tag_api.scopes.get_default()
    click.echo(
        json.dumps(
            {
                "session_id": session.session_id,
                "scope_id": scope.scope_id,
                "status": str(session.status),
            },
            indent=2,
            sort_keys=True,
        )
    )


@research.command("smoke")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def research_smoke(api_key: str | None, backend_url: str | None) -> None:
    """Run limits + Tag smoke in one command."""
    ctx = click.get_current_context()
    ctx.invoke(limits_get, api_key=api_key, backend_url=backend_url)
    click.echo("")
    ctx.invoke(tag_smoke, request="Research SDK smoke", api_key=api_key, backend_url=backend_url)
