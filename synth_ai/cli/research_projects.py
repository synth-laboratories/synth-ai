"""Thin CLI adapters for stable Research project data resources."""

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


def _metadata(value: str | None):
    from synth_ai.research import ResourceMetadata

    if value is None:
        return ResourceMetadata()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise click.BadParameter("metadata must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise click.BadParameter("metadata must be a JSON object")
    return ResourceMetadata(decoded)


def _echo_json(value: object) -> None:
    click.echo(json.dumps(value, indent=2, sort_keys=True))


def _client(api_key: str | None, backend_url: str | None):
    from synth_ai import SynthClient

    return SynthClient(
        api_key=_resolve_api_key(api_key),
        base_url=_resolve_backend_url(backend_url),
    )


_AUTH_OPTIONS = (
    click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Backend base URL."),
    click.option("--api-key", envvar="SYNTH_API_KEY", help="Synth API key."),
)


def _auth_options(function):
    for option in _AUTH_OPTIONS:
        function = option(function)
    return function


@click.group()
def projects() -> None:
    """Manage stable Research project resources."""


@projects.group()
def repositories() -> None:
    """Manage external project repositories."""


@repositories.command("list")
@click.argument("project_id")
@_auth_options
def repositories_list(
    project_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """List external repositories attached to PROJECT_ID."""
    from synth_ai.research import ProjectId

    with _client(api_key, backend_url) as client:
        rows = client.research.projects.repositories.list(ProjectId(project_id))
        _echo_json([row.to_wire() for row in rows])


@repositories.command("create")
@click.argument("project_id")
@click.option("--name", required=True, help="Repository display name.")
@click.option("--url", required=True, help="GitHub repository URL.")
@click.option("--default-branch", help="Default branch to clone.")
@click.option(
    "--role",
    type=click.Choice(("primary", "dependency")),
    default="dependency",
    show_default=True,
)
@click.option("--metadata-json", help="Immutable resource metadata JSON object.")
@_auth_options
def repositories_create(
    project_id: str,
    name: str,
    url: str,
    default_branch: str | None,
    role: str,
    metadata_json: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Attach one external repository to PROJECT_ID."""
    from synth_ai.research import (
        ProjectId,
        ProjectRepositoryRole,
        ProjectRepositorySpec,
    )

    request = ProjectRepositorySpec(
        name=name,
        url=url,
        default_branch=default_branch,
        role=ProjectRepositoryRole(role),
        metadata=_metadata(metadata_json),
    )
    with _client(api_key, backend_url) as client:
        row = client.research.projects.repositories.create(
            ProjectId(project_id),
            request,
        )
        _echo_json(row.to_wire())


@repositories.command("update")
@click.argument("project_id")
@click.argument("repository_id")
@click.option("--url", help="Replacement GitHub repository URL.")
@click.option("--default-branch", help="Replacement default branch.")
@click.option("--role", type=click.Choice(("primary", "dependency")))
@click.option("--metadata-json", help="Replace resource metadata with this JSON object.")
@_auth_options
def repositories_update(
    project_id: str,
    repository_id: str,
    url: str | None,
    default_branch: str | None,
    role: str | None,
    metadata_json: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Update one external project repository."""
    from synth_ai.research import (
        ProjectId,
        ProjectRepositoryId,
        ProjectRepositoryPatch,
        ProjectRepositoryRole,
    )

    request = ProjectRepositoryPatch(
        url=url,
        default_branch=default_branch,
        role=ProjectRepositoryRole(role) if role is not None else None,
        metadata=_metadata(metadata_json) if metadata_json is not None else None,
    )
    with _client(api_key, backend_url) as client:
        row = client.research.projects.repositories.update(
            ProjectId(project_id),
            ProjectRepositoryId(repository_id),
            request,
        )
        _echo_json(row.to_wire())


@repositories.command("delete")
@click.argument("project_id")
@click.argument("repository_id")
@_auth_options
def repositories_delete(
    project_id: str,
    repository_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Delete one external project repository."""
    from synth_ai.research import ProjectId, ProjectRepositoryId

    with _client(api_key, backend_url) as client:
        receipt = client.research.projects.repositories.delete(
            ProjectId(project_id),
            ProjectRepositoryId(repository_id),
        )
        _echo_json(receipt.to_wire())


@projects.group()
def datasets() -> None:
    """Manage project datasets."""


@datasets.command("list")
@click.argument("project_id")
@_auth_options
def datasets_list(
    project_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """List datasets attached to PROJECT_ID."""
    from synth_ai.research import ProjectId

    with _client(api_key, backend_url) as client:
        rows = client.research.projects.datasets.list(ProjectId(project_id))
        _echo_json([row.to_wire() for row in rows])


@datasets.command("upload")
@click.argument("project_id")
@click.argument(
    "source",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option("--name", help="Dataset name; defaults to the source filename.")
@click.option("--content-type", help="Dataset MIME type.")
@click.option("--format", "format_value", help="Dataset format label.")
@click.option("--row-count", type=click.IntRange(min=0))
@click.option("--metadata-json", help="Immutable resource metadata JSON object.")
@_auth_options
def datasets_upload(
    project_id: str,
    source: Path,
    name: str | None,
    content_type: str | None,
    format_value: str | None,
    row_count: int | None,
    metadata_json: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Upload SOURCE as a binary-safe project dataset."""
    from synth_ai.research import ProjectDatasetUpload, ProjectId

    request = ProjectDatasetUpload.from_bytes(
        name=name or source.name,
        content=source.read_bytes(),
        content_type=content_type,
        format=format_value,
        row_count=row_count,
        metadata=_metadata(metadata_json),
    )
    with _client(api_key, backend_url) as client:
        row = client.research.projects.datasets.upload(ProjectId(project_id), request)
        _echo_json(row.to_wire())


@datasets.command("download")
@click.argument("project_id")
@click.argument("dataset_id")
@click.argument("destination", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Replace DESTINATION when it exists.")
@_auth_options
def datasets_download(
    project_id: str,
    dataset_id: str,
    destination: Path,
    force: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    """Download one project dataset to DESTINATION without text coercion."""
    from synth_ai.research import ProjectDatasetId, ProjectId

    if destination.exists() and not force:
        raise click.ClickException(
            f"destination already exists: {destination} (pass --force to replace it)"
        )
    with _client(api_key, backend_url) as client:
        content = client.research.projects.datasets.download(
            ProjectId(project_id),
            ProjectDatasetId(dataset_id),
        )
    destination.write_bytes(content)
    _echo_json({"destination": str(destination), "size_bytes": len(content)})


__all__ = ["projects"]
