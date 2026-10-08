"""Private Synth Index research intake commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def index() -> None:
    """Search Index and submit private research contributions."""


def _credentials(api_key: str | None) -> str:
    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required")
    return api_key


def _emit(value: object) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _limits(mode: str):
    if mode == "fast":
        return None
    from synth_ai.sdk.index.search import SearchExecutionLimits

    return SearchExecutionLimits()


@index.command("search")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="fast", show_default=True)
@click.option("--timeout-seconds", type=click.FloatRange(min=0), default=120.0, show_default=True)
@click.option("--idempotency-key", help="Stable retry identity for this logical search.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
def search(
    query: str,
    mode: str,
    timeout_seconds: float,
    idempotency_key: str | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Run a fast search or create and wait for a durable deep search."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.client import SearchExecutionError, SearchWaitTimeoutError
    from synth_ai.sdk.index.search import SearchSpec

    spec = SearchSpec(query=query, mode=mode, limits=_limits(mode))
    try:
        with SynthClient(api_key=_credentials(api_key), base_url=backend_url) as client:
            if mode == "fast":
                result = client.index.search(spec, idempotency_key=idempotency_key)
            else:
                handle = client.index.searches.create(
                    spec, idempotency_key=idempotency_key
                )
                result = handle.wait(timeout_seconds=timeout_seconds)
    except SearchWaitTimeoutError as error:
        raise click.ClickException(
            f"wait timed out; search remains available as {error.search_id}"
        ) from error
    except (ValueError, SynthError, HTTPError, SearchExecutionError) as error:
        raise click.ClickException(str(error)) from error
    _emit(result)


@index.group("searches")
def searches() -> None:
    """Create, reconnect to, inspect, and cancel durable searches."""


@searches.command("create")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="deep", show_default=True)
@click.option("--idempotency-key", help="Stable retry identity for this logical search.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key.")
def searches_create(
    query: str,
    mode: str,
    idempotency_key: str | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Create a durable search and print its stable identity and state."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import SearchSpec

    try:
        spec = SearchSpec(query=query, mode=mode, limits=_limits(mode))
        with SynthClient(api_key=_credentials(api_key), base_url=backend_url) as client:
            handle = client.index.searches.create(
                spec, idempotency_key=idempotency_key
            )
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    _emit(handle.snapshot)


def _search_read(
    action: str,
    search_id: str,
    api_key: str | None,
    backend_url: str | None,
    *,
    after: int = 0,
    limit: int = 200,
) -> None:
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError

    try:
        with SynthClient(api_key=_credentials(api_key), base_url=backend_url) as client:
            resource = client.index.searches
            if action == "events":
                value = resource.events(search_id, after=after, limit=limit)
            else:
                value = getattr(resource, action)(search_id)
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    _emit(value)


def _search_identity_options(function):
    function = click.option(
        "--api-key", envvar="SYNTH_API_KEY", help="Synth API key."
    )(function)
    return click.option(
        "--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL."
    )(function)


@searches.command("get")
@click.argument("search_id")
@_search_identity_options
def searches_get(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Read current state without resubmitting execution."""
    _search_read("get", search_id, api_key, backend_url)


@searches.command("result")
@click.argument("search_id")
@_search_identity_options
def searches_result(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Read an available terminal result."""
    _search_read("result", search_id, api_key, backend_url)


@searches.command("events")
@click.argument("search_id")
@click.option("--after", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--limit", type=click.IntRange(min=1, max=200), default=200, show_default=True)
@_search_identity_options
def searches_events(
    search_id: str,
    after: int,
    limit: int,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Read ordered reconnectable progress events."""
    _search_read("events", search_id, api_key, backend_url, after=after, limit=limit)


@searches.command("cancel")
@click.argument("search_id")
@_search_identity_options
def searches_cancel(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Request cancellation of a queued or running search."""
    _search_read("cancel", search_id, api_key, backend_url)


@index.group()
def research() -> None:
    """Intake a verified research export conversion."""


@research.command("preview")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
def preview(conversion: Path) -> None:
    """Check exact package bytes and show provenance, evidence and review state."""
    from synth_ai.sdk.index.research_intake import preview_conversion

    try:
        _, _, summary = preview_conversion(conversion)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


@research.command("submit")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--state-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Persistent retry identity; defaults beside the conversion package.",
)
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option(
    "--api-key", envvar="SYNTH_API_KEY", help="Synth API key from an authorized local environment."
)
@click.option(
    "--finalize-only",
    is_flag=True,
    help="Upload and finalize a private draft while holding submission for a separate review decision.",
)
def submit(
    conversion: Path,
    state_file: Path | None,
    backend_url: str | None,
    api_key: str | None,
    finalize_only: bool,
) -> None:
    """Resume private intake; the server controls review eligibility."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.research_intake import (
        IntakeLocked,
        IntakeStateError,
        TerminalRevision,
        submit_conversion,
    )

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required for research intake")
    state_path = state_file or conversion / ".research-intake-state.json"
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            result = submit_conversion(
                client.index, conversion, state_path, finalize_only=finalize_only
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SynthError,
        HTTPError,
        IntakeLocked,
        IntakeStateError,
        TerminalRevision,
    ) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))
