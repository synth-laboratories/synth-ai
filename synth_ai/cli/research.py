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
    """Typed Research projects, swarms, and factories."""


@research.group()
def limits() -> None:
    """Org limits readout."""


@limits.command("get")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def limits_get(api_key: str | None, backend_url: str | None) -> None:
    """Fetch the advanced organization limits projection."""
    from synth_ai import SynthClient

    client = SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )
    payload = client.research.advanced.limits.retrieve()
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@research.group()
def tag() -> None:
    """Factory Tag smoke loop."""


@tag.command("smoke")
@click.option(
    "--request", default="Summarize factory status", show_default=True, help="Tag session request."
)
@click.option("--factory-id", required=True, help="Owning Factory ID.")
@click.option("--effort-id", required=True, help="Owning canonical-project Effort ID.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def tag_smoke(
    request: str,
    factory_id: str,
    effort_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Create a Tag session, send a message, and print scope metadata."""
    from synth_ai import SynthClient
    from synth_ai.core.research._legacy.models.tag import TagSessionCreateRequest

    client = SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )
    tag_api = client.research.advanced.tag
    session = tag_api.create_session(
        TagSessionCreateRequest(
            request=request,
            factory_id=factory_id,
            effort_id=effort_id,
        )
    )
    tag_api.send_message(session.session_id, "Smoke check-in")
    scope = tag_api.get_default_scope()
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
@click.option("--factory-id", required=True, help="Owning Factory ID.")
@click.option("--effort-id", required=True, help="Owning canonical-project Effort ID.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def research_smoke(
    factory_id: str,
    effort_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Run limits + Tag smoke in one command."""
    ctx = click.get_current_context()
    ctx.invoke(limits_get, api_key=api_key, backend_url=backend_url)
    click.echo("")
    ctx.invoke(
        tag_smoke,
        request="Research SDK smoke",
        factory_id=factory_id,
        effort_id=effort_id,
        api_key=api_key,
        backend_url=backend_url,
    )


@research.group()
def swarms() -> None:
    """Create and observe stable Research swarms."""


@swarms.command("start")
@click.option("--objective", required=True, help="Bounded objective for the swarm.")
@click.option("--timebox-seconds", type=int, default=900, show_default=True)
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def swarms_start(
    objective: str,
    timebox_seconds: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Create one swarm and print its durable identity."""
    from synth_ai import SynthClient
    from synth_ai.research import SwarmSpec

    with SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    ) as client:
        handle = client.research.swarms.create(
            SwarmSpec(
                objective=objective,
                timebox_seconds=timebox_seconds,
            )
        )
        click.echo(
            json.dumps(
                {
                    "swarm_id": handle.swarm_id,
                    "state": handle.initial.state.value,
                },
                indent=2,
                sort_keys=True,
            )
        )


@research.group()
def factories() -> None:
    """Inspect stable Research Factories."""


@factories.command("list")
@click.option("--include-archived", is_flag=True)
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL.")
def factories_list(
    include_archived: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """List Factories through the bounded operation contract."""
    from synth_ai import SynthClient

    with SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    ) as client:
        payload = [
            {
                "factory_id": factory.factory_id,
                "name": factory.name,
                "state": factory.state.value,
            }
            for factory in client.research.factories.list(
                include_archived=include_archived
            )
        ]
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
