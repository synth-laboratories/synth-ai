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
    type=click.IntRange(1, 90),
    default=90,
    show_default=True,
    help="Deep execution ceiling; ignored for fast only by omitting it from the request.",
)
@click.option(
    "--private-collection",
    "private_collections",
    multiple=True,
    help="Search only these authorized private collection IDs.",
)
@click.option("--idempotency-key", help="Stable retry identity for this logical search.")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Authorized Synth API key.")
def search(
    query: str,
    mode: str,
    max_results: int,
    deadline_seconds: int,
    private_collections: tuple[str, ...],
    idempotency_key: str | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Return exact Contribution evidence using fast or durable deep retrieval."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import (
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
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Return a cited answer or an explicit insufficient-evidence result."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.search import (
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
                idempotency_key=idempotency_key,
            )
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


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
