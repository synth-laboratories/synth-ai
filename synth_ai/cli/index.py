"""Private Synth Index research intake commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def index() -> None:
    """Search, inspect and submit Synth Index research Contributions."""


@index.command("search")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="fast", show_default=True)
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option(
    "--deadline-seconds",
    type=click.IntRange(1, 300),
    default=180,
    show_default=True,
    help="Durable Deep execution ceiling; omitted from fast requests.",
)
@click.option(
    "--private-collection",
    "private_collections",
    multiple=True,
    help="Search only these authorized private collection IDs.",
)
@click.option("--idempotency-key", help="Stable retry identity for this logical search.")
@click.option("--allow-wallet", is_flag=True, help="Explicitly allow wallet funding.")
@click.option("--max-charge-cents", type=click.IntRange(0, 1_000_000), help="Maximum charge in cents.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Authorized Synth API key.")
def search(
    query: str,
    mode: str,
    max_results: int,
    deadline_seconds: int,
    private_collections: tuple[str, ...],
    idempotency_key: str | None,
    allow_wallet: bool,
    max_charge_cents: int | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Return exact Contribution evidence using fast or durable deep retrieval."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import (
        SearchBillingConstraints,
        SearchExecutionCancelledError,
        SearchExecutionFailedError,
        SearchExecutionLimits,
        SearchMode,
        SearchScope,
        SearchWaitTimeoutError,
    )

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required for Index search")
    selected_mode = SearchMode(mode)
    scope = (
        SearchScope(visibility="private", collection_ids=private_collections)
        if private_collections
        else SearchScope()
    )
    limits = (
        SearchExecutionLimits(deadline_seconds=deadline_seconds)
        if selected_mode == SearchMode.DEEP
        else None
    )
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            result = client.index.search(
                query=query,
                mode=selected_mode,
                scope=scope,
                max_results=max_results,
                limits=limits,
                billing=SearchBillingConstraints(
                    allow_wallet=allow_wallet, max_charge_cents=max_charge_cents
                ),
                idempotency_key=idempotency_key,
            )
    except (
        ValueError,
        SynthError,
        HTTPError,
        SearchWaitTimeoutError,
        SearchExecutionFailedError,
        SearchExecutionCancelledError,
    ) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@index.command("answer")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="fast", show_default=True)
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option(
    "--max-answer-tokens",
    type=click.IntRange(64, 4096),
    default=1024,
    show_default=True,
)
@click.option(
    "--max-answer-cost-usd-micros",
    type=click.IntRange(min=1),
    help="Caller ceiling for answer admission and synthesis inference.",
)
@click.option(
    "--deadline-seconds",
    type=click.IntRange(1, 90),
    default=90,
    show_default=True,
    help="Deep retrieval ceiling; omitted from fast requests.",
)
@click.option(
    "--private-collection",
    "private_collections",
    multiple=True,
    help="Answer only from these authorized private collection IDs.",
)
@click.option(
    "--idempotency-key",
    required=True,
    help="Stable retry identity for retrieval, admission and synthesis.",
)
@click.option("--allow-wallet", is_flag=True, help="Explicitly allow wallet funding for retrieval.")
@click.option("--max-charge-cents", type=click.IntRange(0, 1_000_000), help="Maximum retrieval charge in cents.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Authorized Synth API key.")
def answer(
    query: str,
    mode: str,
    max_results: int,
    max_answer_tokens: int,
    max_answer_cost_usd_micros: int | None,
    deadline_seconds: int,
    private_collections: tuple[str, ...],
    idempotency_key: str,
    allow_wallet: bool,
    max_charge_cents: int | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Return a cited answer or an explicit insufficient-evidence result."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import (
        SearchBillingConstraints,
        SearchExecutionLimits,
        SearchMode,
        SearchScope,
    )

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required for Index answer")
    selected_mode = SearchMode(mode)
    scope = (
        SearchScope(visibility="private", collection_ids=private_collections)
        if private_collections
        else SearchScope()
    )
    limits = (
        SearchExecutionLimits(deadline_seconds=deadline_seconds)
        if selected_mode == SearchMode.DEEP
        else None
    )
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            result = client.index.answer(
                query=query,
                mode=selected_mode,
                scope=scope,
                max_results=max_results,
                limits=limits,
                max_answer_tokens=max_answer_tokens,
                max_answer_cost_usd_micros=max_answer_cost_usd_micros,
                billing=SearchBillingConstraints(
                    allow_wallet=allow_wallet, max_charge_cents=max_charge_cents
                ),
                idempotency_key=idempotency_key,
            )
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@index.group("searches")
def searches() -> None:
    """Create, reconnect to, inspect, and cancel a durable Search.

    # See: docs/drafts/synth-index-api-design-2026-09-12.md
    """


@searches.command("create")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="deep", show_default=True)
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option("--deadline-seconds", type=click.IntRange(1, 300), default=180, show_default=True)
@click.option("--private-collection", "private_collections", multiple=True)
@click.option(
    "--idempotency-key", required=True, help="Reuse this key after an uncertain response."
)
@click.option("--allow-wallet", is_flag=True, help="Explicitly allow wallet funding.")
@click.option("--max-charge-cents", type=click.IntRange(0, 1_000_000), help="Maximum charge in cents.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
@click.option("--api-key", envvar="SYNTH_API_KEY")
def searches_create(
    query: str,
    mode: str,
    max_results: int,
    deadline_seconds: int,
    private_collections: tuple[str, ...],
    idempotency_key: str,
    allow_wallet: bool,
    max_charge_cents: int | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Create one Search and print its durable identity before waiting."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import (
        SearchBillingConstraints,
        SearchContent,
        SearchExecutionLimits,
        SearchMode,
        SearchScope,
        SearchSpec,
    )

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required")
    selected_mode = SearchMode(mode)
    try:
        spec = SearchSpec(
            query=query,
            mode=selected_mode,
            content=SearchContent(max_results=max_results),
            scope=(
                SearchScope(visibility="private", collection_ids=private_collections)
                if private_collections
                else SearchScope()
            ),
            limits=(
                SearchExecutionLimits(deadline_seconds=deadline_seconds)
                if selected_mode is SearchMode.DEEP
                else None
            ),
            billing=SearchBillingConstraints(
                allow_wallet=allow_wallet, max_charge_cents=max_charge_cents
            ),
        )
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            snapshot = client.index.searches.create(spec, idempotency_key=idempotency_key).snapshot
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))


def _read_search(
    operation: str,
    search_id: str,
    backend_url: str | None,
    api_key: str | None,
    *,
    after: int = 0,
    limit: int = 200,
) -> None:
    """Read the same durable identity; result validation uses its stored spec."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required")
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            resource = client.index.searches
            if operation == "get":
                value = resource.get(search_id)
            elif operation == "result":
                snapshot = resource.get(search_id)
                value = resource.result(search_id, snapshot.spec)
            elif operation == "events":
                value = resource.events(search_id, after=after, limit=limit)
            elif operation == "cancel":
                value = resource.cancel(search_id)
            else:
                raise AssertionError(f"Unknown Search read operation: {operation}")
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True))


def _search_identity_options(command):
    command = click.option("--api-key", envvar="SYNTH_API_KEY")(command)
    return click.option("--backend-url", envvar="SYNTH_BACKEND_URL")(command)


@searches.command("get")
@click.argument("search_id")
@_search_identity_options
def searches_get(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Read current Search state without resubmitting it."""
    _read_search("get", search_id, backend_url, api_key)


@searches.command("result")
@click.argument("search_id")
@_search_identity_options
def searches_result(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Read a completed Search result under current authorization."""
    _read_search("result", search_id, backend_url, api_key)


@searches.command("events")
@click.argument("search_id")
@click.option("--after", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--limit", type=click.IntRange(1, 200), default=200, show_default=True)
@_search_identity_options
def searches_events(
    search_id: str,
    after: int,
    limit: int,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Page ordered Search progress from a reconnectable cursor."""
    _read_search("events", search_id, backend_url, api_key, after=after, limit=limit)


@searches.command("cancel")
@click.argument("search_id")
@_search_identity_options
def searches_cancel(search_id: str, backend_url: str | None, api_key: str | None) -> None:
    """Request durable cancellation; already incurred usage remains recorded."""
    _read_search("cancel", search_id, backend_url, api_key)


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
