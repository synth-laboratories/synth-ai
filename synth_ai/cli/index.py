"""Synth Index CLI: free public Search, keyed (paid) Search and research intake.

# See: synth_ai/sdk/index/README.md (Public search / CLI route selection)
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import click
from click.core import ParameterSource

if TYPE_CHECKING:
    from synth_ai.sdk.index.search import SearchFilters


@click.group()
def index() -> None:
    """Search the Synth Index (public or keyed) and run authorized research intake.

    `index search --public` is the free anonymous public route (no API key is
    sent). Keyed Search (`--keyed`, or an API key) is paid and org-funded.
    """


class SearchRoute(StrEnum):
    """Which backend Search route a CLI call uses; never switched implicitly."""

    PUBLIC = "public"
    KEYED = "keyed"


#: Options that only the keyed (paid) route accepts, by CLI parameter name.
KEYED_ONLY_OPTIONS: dict[str, str] = {
    "private_collections": "--private-collection",
    "allow_wallet": "--allow-wallet",
    "max_charge_cents": "--max-charge-cents",
    "deadline_seconds": "--deadline-seconds",
}

ROUTE_REQUIRED_MESSAGE = (
    "Choose a Search route: pass --public for the free anonymous public Search "
    "(no API key), or set SYNTH_API_KEY / --api-key and pass --keyed for paid keyed Search."
)


def _given(context: click.Context, name: str) -> bool:
    """True when the caller set ``name`` on the command line (not a default or env)."""
    return context.get_parameter_source(name) is ParameterSource.COMMANDLINE


def resolve_search_route(
    context: click.Context,
    *,
    public: bool,
    keyed: bool,
    api_key: str | None,
) -> SearchRoute:
    """Pick the Search route from explicit intent; refuse conflicts instead of guessing.

    The SDK chooses a route by client type (``PublicIndexClient`` is anonymous;
    ``SynthClient().index`` is keyed and gets 409 from the public route), so the
    CLI never infers one route from the other. An inherited ``SYNTH_API_KEY``
    does not move ``--public`` onto the paid route (the key is not sent), and a
    keyed request without a key is refused rather than downgraded to public.
    """
    if public and keyed:
        raise click.UsageError("--public and --keyed are mutually exclusive.")
    keyed_options = [flag for name, flag in KEYED_ONLY_OPTIONS.items() if _given(context, name)]
    api_key_source = context.get_parameter_source("api_key")
    if public:
        if api_key_source is ParameterSource.COMMANDLINE:
            raise click.UsageError(
                "--public is the anonymous route and never sends a key; drop --api-key, "
                "or use --keyed for paid keyed Search."
            )
        if keyed_options:
            raise click.UsageError(
                f"{', '.join(keyed_options)} {'is' if len(keyed_options) == 1 else 'are'} "
                "only for paid keyed Search; "
                "drop them or use --keyed instead of --public."
            )
        if api_key:
            click.echo(
                "note: SYNTH_API_KEY is set but not sent; --public uses the free anonymous route.",
                err=True,
            )
        return SearchRoute.PUBLIC
    if not api_key:
        if keyed or keyed_options:
            raise click.UsageError(
                "Keyed Search requires SYNTH_API_KEY or --api-key; it is never "
                "downgraded to the public route. Use --public for free anonymous Search."
            )
        raise click.UsageError(ROUTE_REQUIRED_MESSAGE)
    explicit = keyed or bool(keyed_options) or api_key_source is ParameterSource.COMMANDLINE
    if not explicit:
        click.echo(
            "note: using paid keyed Search because SYNTH_API_KEY is set; pass --keyed to "
            "confirm, or --public for the free anonymous route.",
            err=True,
        )
    return SearchRoute.KEYED


def open_public_index_client(backend_url: str | None):
    """Anonymous client for the public route; carries no credential."""
    from synth_ai.sdk.index import PublicIndexClient

    return PublicIndexClient(base_url=backend_url)


def open_keyed_client(api_key: str, backend_url: str | None):
    """Keyed client for the paid Search route."""
    from synth_ai import SynthClient

    return SynthClient(api_key=api_key, base_url=backend_url)


def _run_public_search(
    query: str,
    *,
    mode: str,
    max_results: int,
    filters: SearchFilters | None,
    idempotency_key: str | None,
    backend_url: str | None,
) -> dict:
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index import (
        PublicSearchDisabledError,
        PublicSearchHandle,
        SearchMode,
        public_search_copy,
        public_search_result_payload,
    )

    selected_mode = SearchMode(mode)
    try:
        with open_public_index_client(backend_url) as client:
            capability = client.public_search_capability()
            if capability is None or not capability.enabled:
                raise PublicSearchDisabledError(
                    "Public Index search is not enabled on this backend", status=404
                )
            if selected_mode not in capability.modes:
                raise PublicSearchDisabledError(
                    f"Public Index search does not offer mode {selected_mode.value!r}",
                    status=404,
                )
            outcome = client.public_search(
                query,
                mode=selected_mode,
                max_results=max_results,
                filters=filters,
                idempotency_key=idempotency_key,
                wait=True,
            )
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    if isinstance(outcome, PublicSearchHandle):  # pragma: no cover - wait=True
        raise click.ClickException("public search returned a handle while waiting")
    return {
        **public_search_result_payload(outcome),
        "terms": public_search_copy(capability).as_dict(),
    }


@index.command("search", short_help="Free public (--public) or paid keyed (--keyed) Search.")
@click.argument("query")
@click.option(
    "--public",
    is_flag=True,
    help="Free anonymous public Search. Sends no API key, even if SYNTH_API_KEY is set.",
)
@click.option(
    "--keyed",
    is_flag=True,
    help="Paid keyed Search with SYNTH_API_KEY / --api-key; never falls back to public.",
)
@click.option("--mode", type=click.Choice(("fast", "deep")), default="fast", show_default=True)
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option("--tag-any", "tags_any", multiple=True, help="Match any selected keyword.")
@click.option("--tag-all", "tags_all", multiple=True, help="Match every selected keyword.")
@click.option("--tag-none", "tags_none", multiple=True, help="Exclude any selected keyword.")
@click.option(
    "--deadline-seconds",
    type=click.IntRange(1, 300),
    default=180,
    show_default=True,
    help="Keyed only: durable Deep execution ceiling; omitted from fast requests.",
)
@click.option(
    "--private-collection",
    "private_collections",
    multiple=True,
    help="Keyed only: search these authorized private collection IDs.",
)
@click.option("--idempotency-key", help="Stable retry identity for this logical search.")
@click.option("--allow-wallet", is_flag=True, help="Keyed only: explicitly allow wallet funding.")
@click.option(
    "--max-charge-cents",
    type=click.IntRange(0, 1_000_000),
    help="Keyed only: maximum charge in cents.",
)
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key (keyed route only).")
@click.pass_context
def search(
    context: click.Context,
    query: str,
    public: bool,
    keyed: bool,
    mode: str,
    max_results: int,
    tags_any: tuple[str, ...],
    tags_all: tuple[str, ...],
    tags_none: tuple[str, ...],
    deadline_seconds: int,
    private_collections: tuple[str, ...],
    idempotency_key: str | None,
    allow_wallet: bool,
    max_charge_cents: int | None,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Search reviewed Contributions over the public or the keyed route.

    --public: free, anonymous, public Contributions only; output includes the
    backend's published `terms`. --keyed (or an API key): paid and org-funded,
    with optional private collections and wallet consent. Choosing no route
    without a key is an error; the CLI never switches routes on its own.
    """
    route = resolve_search_route(context, public=public, keyed=keyed, api_key=api_key)
    from synth_ai.sdk.index.search import SearchFilters

    try:
        filters = SearchFilters(tags_any=tags_any, tags_all=tags_all, tags_none=tags_none)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if route is SearchRoute.PUBLIC:
        payload = _run_public_search(
            query,
            mode=mode,
            max_results=max_results,
            filters=filters if tags_any or tags_all or tags_none else None,
            idempotency_key=idempotency_key,
            backend_url=backend_url,
        )
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    from httpx import HTTPError

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

    assert api_key is not None  # resolve_search_route refuses keyed without a key
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
        with open_keyed_client(api_key, backend_url) as client:
            result = client.index.search(
                query=query,
                mode=selected_mode,
                scope=scope,
                filters=filters,
                max_results=max_results,
                limits=limits,
                billing=SearchBillingConstraints(
                    allow_wallet=allow_wallet, max_charge_cents=max_charge_cents
                ),
                idempotency_key=idempotency_key,
            )
    except (
        SearchWaitTimeoutError,
        SearchExecutionFailedError,
        SearchExecutionCancelledError,
    ) as error:
        click.echo(
            json.dumps(
                {
                    "search_id": error.search_id,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            err=True,
        )
        raise click.ClickException(str(error)) from error
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        json.dumps({"route": "keyed", **result.model_dump(mode="json")}, indent=2, sort_keys=True)
    )


@index.group("searches")
def searches() -> None:
    """Keyed (paid) durable Search: create, reconnect, inspect, cancel.

    Requires SYNTH_API_KEY or --api-key. Public Deep runs through
    `index search --public --mode deep` instead.
    """
    # See: docs/drafts/synth-index-api-design-2026-09-12.md


@searches.command("create")
@click.argument("query")
@click.option("--mode", type=click.Choice(("fast", "deep")), default="deep", show_default=True)
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option("--tag-any", "tags_any", multiple=True, help="Match any selected keyword.")
@click.option("--tag-all", "tags_all", multiple=True, help="Match every selected keyword.")
@click.option("--tag-none", "tags_none", multiple=True, help="Exclude any selected keyword.")
@click.option("--deadline-seconds", type=click.IntRange(1, 300), default=180, show_default=True)
@click.option("--private-collection", "private_collections", multiple=True)
@click.option(
    "--idempotency-key", required=True, help="Reuse this key after an uncertain response."
)
@click.option("--allow-wallet", is_flag=True, help="Explicitly allow wallet funding.")
@click.option(
    "--max-charge-cents", type=click.IntRange(0, 1_000_000), help="Maximum charge in cents."
)
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
@click.option("--api-key", envvar="SYNTH_API_KEY")
def searches_create(
    query: str,
    mode: str,
    max_results: int,
    tags_any: tuple[str, ...],
    tags_all: tuple[str, ...],
    tags_none: tuple[str, ...],
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
        SearchFilters,
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
            filters=SearchFilters(tags_any=tags_any, tags_all=tags_all, tags_none=tags_none),
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


@searches.command("wait")
@click.argument("search_id")
@click.option("--timeout-seconds", type=click.FloatRange(min=0.1), default=180.0, show_default=True)
@_search_identity_options
def searches_wait(
    search_id: str, timeout_seconds: float, backend_url: str | None, api_key: str | None
) -> None:
    """Reconnect to an existing Search and wait for its result; never resubmits."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.client import SearchHandle
    from synth_ai.sdk.index.search import (
        SearchExecutionCancelledError,
        SearchExecutionFailedError,
        SearchWaitTimeoutError,
    )

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required")
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            resource = client.index.searches
            handle = SearchHandle(resource, resource.get(search_id))
            result = handle.wait(timeout_seconds=timeout_seconds)
    except (
        SearchWaitTimeoutError,
        SearchExecutionFailedError,
        SearchExecutionCancelledError,
    ) as error:
        click.echo(
            json.dumps(
                {
                    "search_id": error.search_id,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            err=True,
        )
        raise click.ClickException(str(error)) from error
    except (ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


# --- Contribution lifecycle and private QA -------------------------------------
# Exit codes: 0 ok; 2 usage; 3 not authenticated; 4 not authorized/forbidden;
# 5 conflict or stale version (re-read, do not blindly retry); 6 retryable
# (rate limit / unavailable: reuse the SAME idempotency key); 7 local/contract
# validation refusal (nothing was sent or the response was rejected); 1 other.
EXIT_AUTH, EXIT_FORBIDDEN, EXIT_CONFLICT, EXIT_RETRYABLE, EXIT_CONTRACT = 3, 4, 5, 6, 7


def index_exit_code(error: BaseException) -> int:
    from synth_ai.core.errors import SynthError, SynthErrorCategory

    if isinstance(error, SynthError) and error.failure is not None:
        return {
            SynthErrorCategory.AUTHENTICATION: EXIT_AUTH,
            SynthErrorCategory.AUTHORIZATION: EXIT_FORBIDDEN,
            SynthErrorCategory.CONFLICT: EXIT_CONFLICT,
            SynthErrorCategory.RATE_LIMITED: EXIT_RETRYABLE,
            SynthErrorCategory.TRANSIENT_SERVICE: EXIT_RETRYABLE,
            SynthErrorCategory.RESOURCE_EXHAUSTED: EXIT_RETRYABLE,
            SynthErrorCategory.VALIDATION: EXIT_CONTRACT,
            SynthErrorCategory.CONTRACT_MISMATCH: EXIT_CONTRACT,
        }.get(error.failure.category, 1)
    return EXIT_CONTRACT if isinstance(error, (ValueError, KeyError, TypeError)) else 1


def _fail(error: BaseException) -> None:
    from synth_ai.core.errors import SynthError

    code = index_exit_code(error)
    label = (
        str(error.error_code)
        if isinstance(error, SynthError) and error.error_code
        else type(error).__name__
    )
    hint = ""
    if code == EXIT_RETRYABLE:
        hint = " Retry with the SAME idempotency key; do not create new work."
    elif code == EXIT_CONFLICT:
        hint = " Re-read current state before deciding; nothing was merged."
    elif code == EXIT_AUTH:
        hint = " Check SYNTH_API_KEY and --backend-url."
    elif code == EXIT_FORBIDDEN:
        from synth_ai.sdk.index.scope_errors import scope_denial

        denial = scope_denial(error)
        if denial is not None:
            hint = " " + denial.message()
    click.echo(f"error[{label}]: {error}{hint}", err=True)
    raise click.exceptions.Exit(code)


def _lifecycle_target_options(command):
    command = click.option(
        "--backend-url",
        envvar="SYNTH_BACKEND_URL",
        required=True,
        help="Explicit backend target; no production fallback.",
    )(command)
    return click.option(
        "--api-key",
        envvar="SYNTH_API_KEY",
        required=True,
        help="Already-authorized Synth API credential; never read from disk implicitly.",
    )(command)


def _idempotency_option(command):
    return click.option(
        "--idempotency-key",
        required=True,
        help="Stable identity for this logical action; reuse it after an uncertain response.",
    )(command)


def _lifecycle_run(api_key, backend_url, action):
    """Run one SDK operation, print its exact JSON result, map failures to exit codes."""
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError

    try:
        with open_keyed_client(api_key, backend_url) as client:
            result = action(client.index)
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    except click.exceptions.Exit:
        raise
    except (OSError, ValueError, KeyError, TypeError, SynthError, HTTPError) as error:
        _fail(error)


def _spec(path: Path, model):
    from synth_ai.sdk.index.research.files import read_contract_file

    return read_contract_file(path, model)


_SPEC_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)


@index.group("contribution")
def contribution() -> None:
    """Private draft, upload, submit, repair, withdraw and publish lifecycle."""


@contribution.command("create")
@_idempotency_option
@_lifecycle_target_options
def contribution_create(idempotency_key, api_key, backend_url):
    """Create one private draft. Never submits or publishes."""
    _lifecycle_run(
        api_key, backend_url, lambda i: i.contributions.create(idempotency_key=idempotency_key)
    )


@contribution.command("status")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.option(
    "--search-id", default=None, help="Delivered Search receipt for testing-corpus reads."
)
@_lifecycle_target_options
def contribution_status(contribution_id, revision_id, search_id, api_key, backend_url):
    """Exact revision status, sealed package and reviewer assessments."""
    from synth_ai.sdk.index.contracts import ContributionReference

    reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.revisions.retrieve(
            reference, **({"search_id": search_id} if search_id is not None else {})
        ),
    )


@contribution.command("upload")
@click.argument("draft_file", type=_SPEC_FILE)
@click.argument("upload_spec_file", type=_SPEC_FILE)
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Explicit directory the listed files are confined to; nothing else is scanned.",
)
@click.option(
    "--file",
    "files",
    multiple=True,
    required=True,
    metavar="LOGICAL=RELATIVE_PATH",
    help="One explicitly selected local file per declared asset; repeat per file.",
)
@_lifecycle_target_options
def contribution_upload(draft_file, upload_spec_file, root, files, api_key, backend_url):
    """Prepare, transfer exactly the selected files, then finalize. Does not submit."""
    from synth_ai.mcp.research.tools.index import read_selected_files
    from synth_ai.sdk.index.contributions import ContributionDraft, ContributionUploadSpec

    selected: dict[str, str] = {}
    for item in files:
        logical, separator, relative = item.partition("=")
        if not separator or not logical or not relative or logical in selected:
            raise click.UsageError(f"--file needs unique LOGICAL=RELATIVE_PATH, got {item!r}")
        selected[logical] = relative

    def action(i):
        # Selected bytes are read (confined, bounded, secret-screened) before any
        # contract parse or request; a refused file never reaches the network.
        content = read_selected_files(str(root), selected)
        draft = _spec(draft_file, ContributionDraft)
        spec = _spec(upload_spec_file, ContributionUploadSpec)
        prepared = i.contributions.prepare_upload(draft, spec)
        i.contributions.upload(prepared, content)
        publication = i.contributions.finalize(draft, prepared)
        return {
            "publication": publication.model_dump(mode="json"),
            "uploaded_paths": sorted(content),
            "next_step": "Submit this finalized revision for independent review.",
        }

    _lifecycle_run(api_key, backend_url, action)


@contribution.command("submit")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_lifecycle_target_options
def contribution_submit(contribution_id, revision_id, spec_file, api_key, backend_url):
    """Submit a finalized revision for review. Does not approve or publish."""
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.submission import ContributionSubmitSpec

    reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.submit(reference, _spec(spec_file, ContributionSubmitSpec)),
    )


@contribution.command("repair")
@click.argument("contribution_id")
@click.argument("parent_revision_id")
@_idempotency_option
@_lifecycle_target_options
def contribution_repair(contribution_id, parent_revision_id, idempotency_key, api_key, backend_url):
    """Open a private child revision of a reviewed parent; the parent stays immutable."""
    from synth_ai.sdk.index.lifecycle import RevisionCreateSpec

    spec = RevisionCreateSpec(parent_revision_id=parent_revision_id)
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.revisions.create(
            contribution_id, spec, idempotency_key=idempotency_key
        ),
    )


@contribution.command("attest-rights")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_lifecycle_target_options
def contribution_attest_rights(contribution_id, revision_id, spec_file, api_key, backend_url):
    """Record the exact sealed revision's rights claim; never approve or publish it."""
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.rights import RightsAttestationSpec

    reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.attest_rights(reference, _spec(spec_file, RightsAttestationSpec)),
    )


@contribution.command("register-correction")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_idempotency_option
@_lifecycle_target_options
def contribution_register_correction(
    contribution_id, revision_id, spec_file, idempotency_key, api_key, backend_url
):
    """Bind a resealed research bundle to a private child; never approve or publish."""
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.contributions import ResearchDraftSpec

    reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.register_research_correction(
            reference, _spec(spec_file, ResearchDraftSpec), idempotency_key=idempotency_key
        ),
    )


@contribution.command("withdraw")
@click.argument("contribution_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_idempotency_option
@_lifecycle_target_options
def contribution_withdraw(contribution_id, spec_file, idempotency_key, api_key, backend_url):
    """Withdraw from Search and new reads; prior downloads cannot be recalled."""
    from synth_ai.sdk.index.lifecycle import WithdrawalSpec

    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.withdraw(
            contribution_id, _spec(spec_file, WithdrawalSpec), idempotency_key=idempotency_key
        ),
    )


@contribution.command("publish")
@click.argument("contribution_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_idempotency_option
@_lifecycle_target_options
def contribution_publish(contribution_id, spec_file, idempotency_key, api_key, backend_url):
    """Publisher only: publish an independently approved revision to its sealed audience."""
    from synth_ai.sdk.index.lifecycle import PublicationSpec

    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.contributions.publish(
            contribution_id, _spec(spec_file, PublicationSpec), idempotency_key=idempotency_key
        ),
    )


@index.group("qa")
def qa() -> None:
    """Private QA cases: conversation, assignments, checks and review recommendations.

    QA state is private coordination, not certified science, public release or a
    reward. Backend QA routes are rollout-gated; an unavailable route is a typed
    error, never a local fallback.
    """


@qa.command("open")
@click.argument("spec_file", type=_SPEC_FILE)
@_lifecycle_target_options
def qa_open(spec_file, api_key, backend_url):
    """Open a QA case for one exact submitted revision."""
    from synth_ai.sdk.index.qa import CreateCaseSpec

    _lifecycle_run(
        api_key, backend_url, lambda i: i.qa.create_case(_spec(spec_file, CreateCaseSpec))
    )


@qa.command("case")
@click.argument("case_id")
@_lifecycle_target_options
def qa_case(case_id, api_key, backend_url):
    """Read case state, version and exact revision."""
    _lifecycle_run(api_key, backend_url, lambda i: i.qa.case(case_id))


@qa.command("events")
@click.argument("case_id")
@click.option("--after", type=click.IntRange(min=0), default=0, show_default=True)
@_lifecycle_target_options
def qa_events(case_id, after, api_key, backend_url):
    """Page the visible conversation, findings and decisions."""
    _lifecycle_run(api_key, backend_url, lambda i: i.qa.events(case_id, after=after))


@qa.command("send")
@click.argument("case_id")
@click.option(
    "--action",
    "action_name",
    required=True,
    type=click.Choice(["message", "respond", "request_changes", "approve", "reject"]),
    help="Case action; the server enforces which role may take it in the current state. "
    "Appeal, escalation and adjudication use their own fenced commands.",
)
@click.option("--expected-version", required=True, type=click.IntRange(min=0))
@click.option("--message", required=True)
@_idempotency_option
@_lifecycle_target_options
def qa_send(case_id, action_name, expected_version, message, idempotency_key, api_key, backend_url):
    """Append one conversation/decision event; a stale version exits 5."""
    from synth_ai.sdk.index.qa import CaseAction, CaseEventSpec

    spec = CaseEventSpec(
        expected_version=expected_version, action=CaseAction(action_name), message=message
    )
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.qa.append_event(case_id, spec, idempotency_key=idempotency_key),
    )


@qa.command("assignments")
@_lifecycle_target_options
def qa_assignments(api_key, backend_url):
    """List your reviewer invitations."""
    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: {"assignments": [a.model_dump(mode="json") for a in i.qa.assignments()]},
    )


@qa.command("accept")
@click.argument("assignment_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_lifecycle_target_options
def qa_accept(assignment_id, spec_file, api_key, backend_url):
    """Accept an assignment with an explicit conflict and provenance declaration."""
    from synth_ai.sdk.index.qa import AcceptAssignmentSpec

    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.qa.accept_assignment(assignment_id, _spec(spec_file, AcceptAssignmentSpec)),
    )


@qa.command("invite")
@click.argument("case_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_lifecycle_target_options
def qa_invite(case_id, spec_file, api_key, backend_url):
    """Coordinator: invite a named reviewer."""
    from synth_ai.sdk.index.qa import AssignmentSpec

    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.qa.invite_reviewer(case_id, _spec(spec_file, AssignmentSpec)),
    )


@qa.command("revoke")
@click.argument("assignment_id")
@_lifecycle_target_options
def qa_revoke(assignment_id, api_key, backend_url):
    """Coordinator: revoke an assignment."""
    _lifecycle_run(api_key, backend_url, lambda i: i.qa.revoke_assignment(assignment_id))


@qa.command("checks")
@click.argument("case_id")
@click.option("--after", type=click.IntRange(min=0), default=0, show_default=True)
@_lifecycle_target_options
def qa_checks(case_id, after, api_key, backend_url):
    """Page recorded check attempts and findings."""
    _lifecycle_run(api_key, backend_url, lambda i: i.qa.checks(case_id, after=after))


@qa.command("reviews")
@click.argument("case_id")
@click.option("--after", type=click.IntRange(min=0), default=0, show_default=True)
@_lifecycle_target_options
def qa_reviews(case_id, after, api_key, backend_url):
    """Page recorded review recommendations."""
    _lifecycle_run(api_key, backend_url, lambda i: i.qa.reviews(case_id, after=after))


def _fenced_options(command):
    for option in reversed(
        (
            click.argument("case_id"),
            click.option("--expected-version", required=True, type=click.IntRange(min=0)),
            click.option(
                "--manifest-digest",
                required=True,
                help="Sealed manifest digest of the case (64 hex) this request was written against.",
            ),
            click.option("--rubric-version", required=True, help="Case rubric version."),
            click.option("--message", required=True),
            _idempotency_option,
            _lifecycle_target_options,
        )
    ):
        command = option(command)
    return command


def _fenced_command(name, helper, spec_name, doc):
    @qa.command(name)
    @_fenced_options
    def command(
        case_id,
        expected_version,
        manifest_digest,
        rubric_version,
        message,
        idempotency_key,
        api_key,
        backend_url,
    ):
        from synth_ai.sdk.index import qa as qa_models

        spec = getattr(qa_models, spec_name)(
            expected_version=expected_version,
            manifest_digest=manifest_digest,
            rubric_version=rubric_version,
            message=message,
        )
        _lifecycle_run(
            api_key,
            backend_url,
            lambda i: getattr(i.qa, helper)(case_id, spec, idempotency_key=idempotency_key),
        )

    command.__doc__ = doc
    return command


_fenced_command(
    "appeal",
    "appeal",
    "AppealSpec",
    "Appeal a rejection or private acceptance (stale fence exits 5).",
)
_fenced_command("escalate", "escalate", "EscalationSpec", "Escalate the case to a coordinator.")
_fenced_command(
    "adjudicate",
    "adjudicate",
    "AdjudicationSpec",
    "Coordinator: independent decision that reopens fresh independent review. Never approves.",
)
_fenced_command(
    "note",
    "add_internal_note",
    "InternalNoteSpec",
    "Reviewer/coordinator: internal note never shown to the contributor.",
)


@qa.command("review")
@click.argument("case_id")
@click.argument("spec_file", type=_SPEC_FILE)
@_idempotency_option
@_lifecycle_target_options
def qa_review(case_id, spec_file, idempotency_key, api_key, backend_url):
    """Record a review recommendation for the exact case revision."""
    from synth_ai.sdk.index.qa_reviews import RecordReviewSpec

    _lifecycle_run(
        api_key,
        backend_url,
        lambda i: i.qa.record_review(
            case_id, _spec(spec_file, RecordReviewSpec), idempotency_key=idempotency_key
        ),
    )


def _annotate_scope_hints() -> None:
    """Append the SDK scope class to each command's help (advisory; backend decides)."""
    from synth_ai.sdk.index.scopes import required_scopes

    hints = {
        (qa, "open"): "index.qa.cases.create",
        (qa, "case"): "index.qa.cases.get",
        (qa, "events"): "index.qa.events.list",
        (qa, "send"): "index.qa.events.create",
        (qa, "appeal"): "index.qa.appeals.create",
        (qa, "escalate"): "index.qa.escalations.create",
        (qa, "note"): "index.qa.notes.create",
        (qa, "adjudicate"): "index.qa.adjudications.create",
        (qa, "assignments"): "index.qa.assignments.list",
        (qa, "accept"): "index.qa.assignments.accept",
        (qa, "invite"): "index.qa.assignments.create",
        (qa, "revoke"): "index.qa.assignments.revoke",
        (qa, "checks"): "index.qa.checks.list",
        (qa, "reviews"): "index.qa.reviews.list",
        (qa, "review"): "index.qa.reviews.record",
        (contribution, "create"): "index.contributions.create",
        (contribution, "upload"): "index.contributions.upload.prepare",
        (contribution, "submit"): "index.contributions.submit",
        (contribution, "repair"): "index.contributions.revisions.create",
        (contribution, "withdraw"): "index.contributions.withdrawal.create",
        (contribution, "publish"): "index.contributions.publication.create",
    }
    for (group, name), operation in hints.items():
        command = group.commands[name]
        scopes = " or ".join(required_scopes(operation))
        command.help = (
            f"{command.help or ''}\n\nOAuth scope (any of): {scopes}. Scopes allow a class "
            "of operation only; the backend decides by role, assignment and ownership."
        )


_annotate_scope_hints()


@index.group()
def research() -> None:
    """Private intake, exact release consent and independent reproduction evidence."""


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
        IntakeLockedError,
        IntakeStateError,
        TerminalRevisionError,
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
        IntakeLockedError,
        IntakeStateError,
        TerminalRevisionError,
    ) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))


def _research_target_options(command):
    command = click.option(
        "--backend-url",
        envvar="SYNTH_BACKEND_URL",
        required=True,
        help="Explicit backend target; no production fallback.",
    )(command)
    command = click.option(
        "--api-key",
        envvar="SYNTH_API_KEY",
        required=True,
        help="Already-authorized Synth API credential.",
    )(command)
    return click.option(
        "--receipt",
        type=click.Path(dir_okay=False, path_type=Path),
        required=True,
        help="Private create-only result file; identical retries are accepted.",
    )(command)


def _research_spec_options(command):
    command = click.argument(
        "spec_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    )(command)
    command = click.argument("revision_id")(command)
    return click.argument("contribution_id")(command)


def _research_operation(
    operation, contribution_id, revision_id, spec_file, backend_url, api_key, receipt
):
    """Validate a typed input, route one explicit operation, then retain its receipt."""
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.research import (
        ReleaseConsentSpec,
        ReproductionAttestationSpec,
        ResearchBindingSpec,
        ResearchRevocationSpec,
    )
    from synth_ai.sdk.index.research.files import read_contract_file, write_private_receipt

    models = {
        "bind": ResearchBindingSpec,
        "consent": ReleaseConsentSpec,
        "attest": ReproductionAttestationSpec,
        "revoke": ResearchRevocationSpec,
    }
    try:
        reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
        spec = read_contract_file(spec_file, models[operation])
        with open_keyed_client(api_key, backend_url) as client:
            result = getattr(client.index.contributions.research, operation)(reference, spec)
        write_private_receipt(receipt, result)
    except (OSError, ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded {operation} receipt: {receipt}")


@research.command("bind")
@_research_spec_options
@_research_target_options
def research_bind(contribution_id, revision_id, spec_file, backend_url, api_key, receipt):
    """Bind a committed private archive to exact approved release bytes."""
    _research_operation(
        "bind", contribution_id, revision_id, spec_file, backend_url, api_key, receipt
    )


@research.command("consent")
@_research_spec_options
@_research_target_options
def research_consent(contribution_id, revision_id, spec_file, backend_url, api_key, receipt):
    """Record explicit author consent for the exact seal, disclosure and audience."""
    _research_operation(
        "consent", contribution_id, revision_id, spec_file, backend_url, api_key, receipt
    )


@research.command("attest")
@_research_spec_options
@_research_target_options
def research_attest(contribution_id, revision_id, spec_file, backend_url, api_key, receipt):
    """Submit independently obtained reproduction evidence; never run an experiment."""
    _research_operation(
        "attest", contribution_id, revision_id, spec_file, backend_url, api_key, receipt
    )


@research.command("revoke")
@_research_spec_options
@_research_target_options
def research_revoke(contribution_id, revision_id, spec_file, backend_url, api_key, receipt):
    """Revoke the exact release disclosure under the backend's current authority."""
    _research_operation(
        "revoke", contribution_id, revision_id, spec_file, backend_url, api_key, receipt
    )


@research.command("allocate-archive")
@click.argument("contribution_id")
@click.argument("spec_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_research_target_options
def research_allocate_archive(contribution_id, spec_file, backend_url, api_key, receipt):
    """Allocate the explicit frozen snapshot's private archive collection."""
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.research import ResearchArchiveAllocationSpec
    from synth_ai.sdk.index.research.files import read_contract_file, write_private_receipt

    try:
        spec = read_contract_file(spec_file, ResearchArchiveAllocationSpec)
        with open_keyed_client(api_key, backend_url) as client:
            result = client.index.contributions.research.allocate_archive(contribution_id, spec)
        write_private_receipt(receipt, result)
    except (OSError, ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded private archive allocation: {receipt}")


@research.command("archive")
@click.argument("contribution_id")
@click.argument("revision_id")
@_research_target_options
def research_archive(contribution_id, revision_id, backend_url, api_key, receipt):
    """Read private frozen evidence with a current, separate archive grant."""
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.research.files import write_private_receipt

    try:
        reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
        with open_keyed_client(api_key, backend_url) as client:
            result = client.index.contributions.research.archive(reference)
        write_private_receipt(receipt, result)
    except (OSError, ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded private frozen evidence: {receipt}")


@index.command("classify")
@_research_spec_options
@click.option(
    "--idempotency-key", required=True, help="Persist and reuse this exact decision identity."
)
@_research_target_options
def classify(
    contribution_id, revision_id, spec_file, idempotency_key, backend_url, api_key, receipt
):
    """Record an independent metadata decision against an exact sealed revision."""
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.classification import ClassificationSpec
    from synth_ai.sdk.index.contracts import ContributionReference
    from synth_ai.sdk.index.research.files import read_contract_file, write_private_receipt

    try:
        reference = ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
        spec = read_contract_file(spec_file, ClassificationSpec)
        with open_keyed_client(api_key, backend_url) as client:
            result = client.index.contributions.classifications.create(
                reference, spec, idempotency_key=idempotency_key
            )
        write_private_receipt(receipt, result)
    except (OSError, ValueError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded classification decision: {receipt}")


def _offline_release_options(command):
    for name in ("binding", "descriptor", "manifest"):
        command = click.option(
            f"--{name}",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
        )(command)
    return command


def _offline_release_operation(
    operation, binding, descriptor, manifest, archive_root=None, out=None
):
    """Run the backend-mirrored bounded reconstruction contract without providers."""
    from synth_ai.sdk.index.manifest import decode_manifest
    from synth_ai.sdk.index.research.build import (
        build_release,
        validate_release_binding,
        verify_archive,
        verify_release,
    )
    from synth_ai.sdk.index.research.contracts import DerivationBinding
    from synth_ai.sdk.index.research.files import read_contract_file, read_input_file

    try:
        binding_contract = read_contract_file(binding, DerivationBinding)
        descriptor_bytes = read_input_file(descriptor)
        manifest_contract = decode_manifest(read_input_file(manifest))
        options = {
            "binding": binding_contract,
            "descriptor": descriptor_bytes,
            "manifest": manifest_contract,
        }
        if operation == "validate-release":
            validate_release_binding(**options)
            objects = verify_archive(archive_root, binding_contract)
            result = {"valid": True, "frozen_object_count": len(objects), "provider_calls": 0}
        elif operation == "verify-release":
            result = verify_release(out, **options)
        else:
            if operation == "reproduce-release" and out.exists():
                raise ValueError("Reproduction requires a fresh output directory")
            result = build_release(archive_root, out, **options)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, sort_keys=True))


@research.command("validate-release")
@_offline_release_options
@click.option(
    "--archive-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def research_validate_release(binding, descriptor, manifest, archive_root):
    """Verify exact frozen inputs and approved outputs offline; grant no publication."""
    _offline_release_operation("validate-release", binding, descriptor, manifest, archive_root)


@research.command("build-release")
@_offline_release_options
@click.option(
    "--archive-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_build_release(binding, descriptor, manifest, archive_root, out):
    """Reconstruct only approved bytes; identical retries verify the prior result."""
    _offline_release_operation("build-release", binding, descriptor, manifest, archive_root, out)


@research.command("reproduce-release")
@_offline_release_options
@click.option(
    "--archive-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_reproduce_release(binding, descriptor, manifest, archive_root, out):
    """Independently reconstruct into a fresh directory; prove artifact scope only."""
    _offline_release_operation(
        "reproduce-release", binding, descriptor, manifest, archive_root, out
    )


@research.command("verify-release")
@_offline_release_options
@click.option("--out", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def research_verify_release(binding, descriptor, manifest, out):
    """Check exact output bytes, manifest and artifact reconstruction receipt."""
    _offline_release_operation("verify-release", binding, descriptor, manifest, out=out)


@research.command("capture-codex-task-read")
@click.option(
    "--native-input", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--thread-id", required=True, help="Exact authorized task identity; no session discovery."
)
@click.option("--captured-at", required=True, help="Explicit RFC3339 timestamp with offset.")
@click.option("--cutoff-at", required=True, help="Explicit capture cutoff with offset.")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_capture_codex_task_read(native_input, thread_id, captured_at, cutoff_at, out):
    """Freeze a selected native Codex read privately, with honest partial-export gaps."""
    from datetime import datetime

    from synth_ai.sdk.index.research.capture import freeze_codex_task_read

    try:
        export = freeze_codex_task_read(
            native_input,
            out,
            expected_thread_id=thread_id,
            captured_at=datetime.fromisoformat(captured_at),
            cutoff_at=datetime.fromisoformat(cutoff_at),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded {export.completeness} native capture: {out}")


@research.command("capture-codex-rollout-prefix")
@click.option(
    "--native-input", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--thread-id", required=True, help="Exact authorized task identity; no session discovery."
)
@click.option("--captured-at", required=True, help="Explicit RFC3339 timestamp with offset.")
@click.option("--cutoff-at", required=True, help="Explicit capture cutoff with offset.")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_capture_codex_rollout_prefix(native_input, thread_id, captured_at, cutoff_at, out):
    """Freeze selected complete native JSONL records through an explicit cutoff."""
    from datetime import datetime

    from synth_ai.sdk.index.research.capture import freeze_codex_rollout_prefix

    try:
        export = freeze_codex_rollout_prefix(
            native_input,
            out,
            expected_thread_id=thread_id,
            captured_at=datetime.fromisoformat(captured_at),
            cutoff_at=datetime.fromisoformat(cutoff_at),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded {export.completeness} native capture: {out}")


@research.command("capture-swarms-evidence")
@click.option(
    "--native-input", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--run-id", required=True, help="Exact authorized SMR run identity.")
@click.option("--project-id", required=True, help="Exact authorized SMR project identity.")
@click.option("--captured-at", required=True, help="Explicit RFC3339 timestamp with offset.")
@click.option("--cutoff-at", required=True, help="Source freshness time with offset.")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_capture_swarms_evidence(native_input, run_id, project_id, captured_at, cutoff_at, out):
    """Freeze a selected bounded Swarms evidence response with declared gaps."""
    from datetime import datetime

    from synth_ai.sdk.index.research.capture import freeze_swarms_evidence

    try:
        export = freeze_swarms_evidence(
            native_input,
            out,
            expected_run_id=run_id,
            expected_project_id=project_id,
            captured_at=datetime.fromisoformat(captured_at),
            cutoff_at=datetime.fromisoformat(cutoff_at),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded {export.completeness} native capture: {out}")


@research.command("capture-mlok-policy")
@click.option(
    "--native-input", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--thread-id", required=True, help="Exact selected mlok policy thread identity.")
@click.option("--captured-at", required=True, help="Explicit RFC3339 timestamp with offset.")
@click.option(
    "--cutoff-at", required=True, help="Equal to capture time; native record has no clock."
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
def research_capture_mlok_policy(native_input, thread_id, captured_at, cutoff_at, out):
    """Freeze selected mlok model context with honest partial-export gaps."""
    from datetime import datetime

    from synth_ai.sdk.index.research.capture import freeze_mlok_policy_capture

    try:
        export = freeze_mlok_policy_capture(
            native_input,
            out,
            expected_thread_id=thread_id,
            captured_at=datetime.fromisoformat(captured_at),
            cutoff_at=datetime.fromisoformat(cutoff_at),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Recorded {export.completeness} native capture: {out}")
