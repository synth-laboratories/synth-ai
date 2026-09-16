"""Synth Index commands: public and private reads, and private research intake.

Which backend operations each command sends is declared in
``synth_ai.sdk.index.surfaces``. Read commands work without a credential against
the public surface; account commands and research intake need one. A key file
(``--api-key-file`` / ``SYNTH_API_KEY_FILE``) is re-read when the backend
rejects the key, so a long intake follows key rotation; a rejected key that did
not change is reported as revoked.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

T = TypeVar("T")

_KEY_HELP = "Synth API key from an authorized local environment."
_KEY_FILE_HELP = "File holding the Synth API key (mode 600); re-read after a rejection."


def _connection(command: Callable[..., Any]) -> Callable[..., Any]:
    command = click.option(
        "--api-key-file",
        envvar="SYNTH_API_KEY_FILE",
        type=click.Path(dir_okay=False, path_type=Path),
        help=_KEY_FILE_HELP,
    )(command)
    command = click.option("--api-key", envvar="SYNTH_API_KEY", help=_KEY_HELP)(command)
    return click.option(
        "--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL."
    )(command)


def _credential(api_key: str | None, api_key_file: Path | None):
    from synth_ai.core.auth.renewal import RenewableCredential, file_source, static_source

    if api_key and api_key_file:
        raise click.UsageError("Pass --api-key or --api-key-file, not both")
    if api_key_file:
        return RenewableCredential(file_source(api_key_file))
    if api_key:
        return RenewableCredential(static_source(api_key, "--api-key / SYNTH_API_KEY"))
    return None


def _failure(error: BaseException) -> click.ClickException:
    from synth_ai.sdk.index.errors import index_error_code

    code = index_error_code(error)
    return click.ClickException(f"{error} [{code.value}]" if code else str(error))


def _handled() -> tuple[type[BaseException], ...]:
    from httpx import HTTPError

    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.research_intake import (
        IntakeLocked,
        IntakeStateError,
        TerminalRevision,
    )

    return (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SynthError,
        HTTPError,
        IntakeLocked,
        IntakeStateError,
        TerminalRevision,
    )


def _with_index(
    backend_url: str | None,
    api_key: str | None,
    api_key_file: Path | None,
    authenticated: Callable[[Any], T],
    public: Callable[[Any], T] | None = None,
) -> T:
    """Run one command on an authenticated client, or the public one without a key."""
    from synth_ai.core.auth.renewal import call_with_renewal

    try:
        credential = _credential(api_key, api_key_file)
        if credential is None:
            if public is None:
                raise click.ClickException(
                    "This command needs SYNTH_API_KEY, --api-key or --api-key-file"
                )
            from synth_ai.sdk.index.public import PublicIndexClient

            with PublicIndexClient(base_url=backend_url) as client:
                return public(client)

        def run(key) -> T:
            from synth_ai import SynthClient

            with SynthClient(api_key=key.value, base_url=backend_url) as client:
                return authenticated(client.index)

        return call_with_renewal(credential, run)
    except click.ClickException:
        raise
    except _handled() as error:
        raise _failure(error) from error


def _echo(value: Any) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def _same(call: Callable[[Any], T]) -> tuple[Callable[[Any], T], Callable[[Any], T]]:
    return call, call


def _reference(contribution_id: str, revision_id: str):
    from synth_ai.sdk.index.contracts import ContributionReference

    try:
        return ContributionReference(contribution_id=contribution_id, revision_id=revision_id)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error


@click.group()
def index() -> None:
    """Search Synth Index and submit private research contributions."""


@index.command("capabilities")
@_connection
def capabilities(backend_url, api_key, api_key_file) -> None:
    """Show supported search modes, visibilities, limits and your capabilities."""
    _echo(_with_index(backend_url, api_key, api_key_file, *_same(lambda c: c.capabilities())))


@index.command("tags")
@_connection
def tags(backend_url, api_key, api_key_file) -> None:
    """Show the tag registry and taxonomy used by search filters."""
    _echo(_with_index(backend_url, api_key, api_key_file, *_same(lambda c: c.tags.list())))


@index.command("search")
@click.argument("query")
@click.option(
    "--private",
    "private_scope",
    is_flag=True,
    help="Search your organization's private Contributions (may be billed).",
)
@click.option("--collection", "collections", multiple=True, help="Private collection ID.")
@click.option("--max-results", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option(
    "--idempotency-key",
    help="Reuse to retry one logical search without a second charge.",
)
@_connection
def search(
    query,
    private_scope,
    collections,
    max_results,
    idempotency_key,
    backend_url,
    api_key,
    api_key_file,
) -> None:
    """Search reviewed Contributions; public unless --private is given."""
    from synth_ai.sdk.index.search import SearchContent, SearchScope, SearchSpec

    if collections and not private_scope:
        raise click.UsageError("--collection selects private scope; add --private")
    try:
        spec = SearchSpec(
            query=query,
            scope=SearchScope(
                visibility="private" if private_scope else "public",
                collection_ids=tuple(collections),
            ),
            content=SearchContent(max_results=max_results),
        )
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    _echo(
        _with_index(
            backend_url,
            api_key,
            api_key_file,
            lambda c: c.search(spec, idempotency_key=idempotency_key),
            None if private_scope else (lambda c: c.search(spec)),
        )
    )


@index.command("contents")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.option("--search-id", help="Search receipt that selected this revision (account only).")
@click.option("--max-bytes", type=click.IntRange(1, 65_536), default=65_536, show_default=True)
@_connection
def contents(
    contribution_id, revision_id, search_id, max_bytes, backend_url, api_key, api_key_file
) -> None:
    """Read the exact text of one revision under current authorization."""
    reference = _reference(contribution_id, revision_id)
    _echo(
        _with_index(
            backend_url,
            api_key,
            api_key_file,
            lambda c: c.contents.retrieve(
                references=[reference], search_id=search_id, max_bytes=max_bytes
            ),
            None
            if search_id
            else (lambda c: c.contents.retrieve(references=[reference], max_bytes=max_bytes)),
        )
    )


@index.command("contribution")
@click.argument("contribution_id")
@_connection
def contribution(contribution_id, backend_url, api_key, api_key_file) -> None:
    """Show one Contribution, its current revision and history."""
    _echo(
        _with_index(
            backend_url,
            api_key,
            api_key_file,
            *_same(lambda c: c.contributions.retrieve(contribution_id)),
        )
    )


@index.command("revision")
@click.argument("contribution_id")
@click.argument("revision_id")
@_connection
def revision(contribution_id, revision_id, backend_url, api_key, api_key_file) -> None:
    """Show one exact revision: status, sealed package, assessments, citation."""
    reference = _reference(contribution_id, revision_id)
    _echo(
        _with_index(
            backend_url,
            api_key,
            api_key_file,
            *_same(lambda c: c.contributions.revisions.retrieve(reference)),
        )
    )


@index.command("asset")
@click.argument("contribution_id")
@click.argument("revision_id")
@click.argument("asset_id")
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="File to write the asset bytes to; must not exist.",
)
@_connection
def asset(
    contribution_id, revision_id, asset_id, output, backend_url, api_key, api_key_file
) -> None:
    """Download one declared asset of an exact revision, to check a citation."""
    from hashlib import sha256

    reference = _reference(contribution_id, revision_id)
    content = _with_index(
        backend_url,
        api_key,
        api_key_file,
        *_same(lambda c: c.contributions.assets.retrieve(reference, asset_id)),
    )
    try:
        with output.open("xb") as handle:
            handle.write(content)
    except OSError as error:
        raise click.ClickException(f"Cannot write {output}: {error}") from error
    _echo(
        {
            "asset_id": asset_id,
            "output": str(output),
            "size_bytes": len(content),
            "digest_sha256": sha256(content).hexdigest(),
        }
    )


@index.command("account")
@_connection
def account(backend_url, api_key, api_key_file) -> None:
    """Show your Index identity, organization and capabilities."""
    _echo(_with_index(backend_url, api_key, api_key_file, lambda c: c.account.retrieve()))


@index.command("my-contributions")
@_connection
def my_contributions(backend_url, api_key, api_key_file) -> None:
    """List your Contributions and each current revision status."""
    _echo(_with_index(backend_url, api_key, api_key_file, lambda c: c.account.contributions()))


@index.command("usage")
@_connection
def usage(backend_url, api_key, api_key_file) -> None:
    """Show your organization's search usage and charges."""
    _echo(_with_index(backend_url, api_key, api_key_file, lambda c: c.account.usage()))


@index.command("promo-credit")
@_connection
def promo_credit(backend_url, api_key, api_key_file) -> None:
    """Show the private-search promotional balance and when it resets."""
    _echo(_with_index(backend_url, api_key, api_key_file, lambda c: c.account.promo_credit()))


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


_STATE_HELP = "Persistent retry identity; defaults beside the conversion package."


def _state_path(conversion: Path, state_file: Path | None) -> Path:
    return state_file or conversion / ".research-intake-state.json"


@research.command("submit")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--state-file", type=click.Path(dir_okay=False, path_type=Path), help=_STATE_HELP)
@_connection
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
    api_key_file: Path | None,
    finalize_only: bool,
) -> None:
    """Resume private intake; the server controls review eligibility.

    Safe to re-run after any interruption, including a rejected (rotated) key:
    the saved state and server reconciliation prevent duplicate allocation,
    upload or submission.
    """
    from synth_ai.sdk.index.research_intake import submit_conversion

    if not (api_key or api_key_file):
        raise click.ClickException(
            "SYNTH_API_KEY, --api-key or --api-key-file is required for research intake"
        )
    state_path = _state_path(conversion, state_file)
    result = _with_index(
        backend_url,
        api_key,
        api_key_file,
        lambda c: submit_conversion(c, conversion, state_path, finalize_only=finalize_only),
    )
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@research.command("recover-state")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--state-file", type=click.Path(dir_okay=False, path_type=Path), help=_STATE_HELP)
@click.option(
    "--backend-url",
    required=True,
    help="The backend the old state was used against; v1 state does not record it.",
)
@click.option("--api-key", envvar="SYNTH_API_KEY", help=_KEY_HELP)
@click.option(
    "--api-key-file",
    envvar="SYNTH_API_KEY_FILE",
    type=click.Path(dir_okay=False, path_type=Path),
    help=_KEY_FILE_HELP,
)
def recover_state(
    conversion: Path,
    state_file: Path | None,
    backend_url: str,
    api_key: str | None,
    api_key_file: Path | None,
) -> None:
    """Convert v1 intake state to v2 without creating a duplicate submission.

    Reads the server's record for the saved draft key under your account and
    rewrites the state file to match (the original is kept as
    <state>.v1-backup). Nothing is allocated, uploaded or submitted. Then run
    `research submit` with the same state file to resume.
    """
    from synth_ai.sdk.index.research_intake import recover_v1_state

    if not (api_key or api_key_file):
        raise click.ClickException("Recovery needs the account that used the old state")
    state_path = _state_path(conversion, state_file)
    result = _with_index(
        backend_url,
        api_key,
        api_key_file,
        lambda c: recover_v1_state(c, conversion, state_path, backend_url=backend_url),
    )
    click.echo(json.dumps(result, indent=2, sort_keys=True))
