"""Thin CLI adapters for the stable Environment catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path

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
        raise click.ClickException(
            "api_key is required (pass --api-key or set SYNTH_API_KEY)"
        )
    return resolved


def _client(api_key: str | None, backend_url: str | None):
    from synth_ai import SynthClient

    return SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )


def _manifest(path: Path):
    from synth_ai.research import EnvironmentManifest

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return EnvironmentManifest.from_input(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise click.ClickException(f"invalid Environment manifest: {error}") from error


def _echo(value: object) -> None:
    click.echo(json.dumps(value, indent=2, sort_keys=True))


_AUTH_OPTIONS = (
    click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL."),
    click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key."),
)


def _auth_options(function):
    for option in _AUTH_OPTIONS:
        function = option(function)
    return function


@click.group()
def environments() -> None:
    """Manage immutable Research Environment manifests."""


@environments.command("list")
@click.option("--limit", type=click.IntRange(1, 500), default=100, show_default=True)
@_auth_options
def environments_list(
    limit: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """List cataloged Environment manifest versions."""
    with _client(api_key, backend_url) as client:
        values = client.research.environments.list(limit=limit)
        _echo([value.to_wire() for value in values])


@environments.command("create")
@click.argument(
    "manifest_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@_auth_options
def environments_create(
    manifest_file: Path,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Catalog the immutable manifest in MANIFEST_FILE."""
    with _client(api_key, backend_url) as client:
        value = client.research.environments.create(_manifest(manifest_file))
        _echo(value.to_wire())


@environments.command("get")
@click.argument("name")
@click.option("--manifest-digest", help="Select one immutable manifest version.")
@_auth_options
def environments_get(
    name: str,
    manifest_digest: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Retrieve one Environment manifest version by NAME."""
    from synth_ai.research import EnvironmentDigest, EnvironmentName

    digest_value = (
        EnvironmentDigest(manifest_digest) if manifest_digest is not None else None
    )
    with _client(api_key, backend_url) as client:
        value = client.research.environments.retrieve(
            EnvironmentName(name),
            manifest_digest=digest_value,
        )
        _echo(value.to_wire())


@environments.command("preflight")
@click.argument("name")
@click.option("--manifest-digest", help="Select one immutable manifest version.")
@_auth_options
def environments_preflight(
    name: str,
    manifest_digest: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Run backend-owned preflight for one Environment version."""
    from synth_ai.research import EnvironmentDigest, EnvironmentName

    digest_value = (
        EnvironmentDigest(manifest_digest) if manifest_digest is not None else None
    )
    with _client(api_key, backend_url) as client:
        value = client.research.environments.preflight(
            EnvironmentName(name),
            manifest_digest=digest_value,
        )
        _echo(value.to_wire())


__all__ = ["environments"]
