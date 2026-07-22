"""Thin CLI adapters for customer image releases."""

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


def _json_file(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise click.ClickException(f"invalid JSON file: {error}") from error


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


@click.group(name="image-releases")
def image_releases() -> None:
    """Manage customer image-release receipts."""


@image_releases.command("list")
@_auth_options
def image_releases_list(api_key: str | None, backend_url: str | None) -> None:
    """List actor runtime image materializations."""
    with _client(api_key, backend_url) as client:
        _echo(client.research.image_releases.list().to_wire())


@image_releases.command("upload-url")
@click.argument(
    "request_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@_auth_options
def image_releases_upload_url(
    request_file: Path,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Create a presigned upload URL from REQUEST_FILE."""
    from synth_ai.research import ImageReleaseUploadRequest

    with _client(api_key, backend_url) as client:
        request = ImageReleaseUploadRequest.from_wire(_json_file(request_file))
        _echo(client.research.image_releases.create_upload(request).to_wire())


@image_releases.command("finalize")
@click.argument(
    "request_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@_auth_options
def image_releases_finalize(
    request_file: Path,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Finalize an uploaded image release from REQUEST_FILE."""
    from synth_ai.research import ImageReleaseFinalizeRequest

    with _client(api_key, backend_url) as client:
        request = ImageReleaseFinalizeRequest.from_wire(_json_file(request_file))
        _echo(client.research.image_releases.finalize(request).to_wire())


@image_releases.command("get")
@click.argument("release_id")
@_auth_options
def image_releases_get(
    release_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Retrieve one immutable image-release receipt."""
    from synth_ai.research import ImageReleaseId

    with _client(api_key, backend_url) as client:
        _echo(client.research.image_releases.retrieve(ImageReleaseId(release_id)).to_wire())


@image_releases.command("archive")
@click.argument("runtime_image_release_id")
@_auth_options
def image_releases_archive(
    runtime_image_release_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Archive one actor runtime image materialization."""
    from synth_ai.research import RuntimeImageReleaseId

    with _client(api_key, backend_url) as client:
        _echo(
            client.research.image_releases.archive(
                RuntimeImageReleaseId(runtime_image_release_id)
            ).to_wire()
        )


__all__ = ["image_releases"]
