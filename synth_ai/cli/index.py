"""Synth Index CLI: free public Search, keyed (paid) Search and research intake.

# See: synth_ai/sdk/index/README.md (Public search / CLI route selection)
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import click
from click.core import ParameterSource


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
    if route is SearchRoute.PUBLIC:
        payload = _run_public_search(
            query,
            mode=mode,
            max_results=max_results,
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
