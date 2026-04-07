"""Containers CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click


def _load_json_argument(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = raw.strip()
    if value.startswith("@"):
        value = Path(value[1:]).read_text()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise click.ClickException("Expected a JSON object.")
    return parsed


def _client(api_key: str | None, backend_url: str | None):
    from synth_ai.sdk.containers import ContainersClient

    return ContainersClient(api_key=api_key, backend_base=backend_url)


def _echo(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.group()
def containers() -> None:
    """Manage hosted containers."""


@containers.command("list")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def list_containers(api_key: str | None, backend_url: str | None) -> None:
    items = _client(api_key, backend_url).list()
    _echo([item.model_dump() for item in items])


@containers.command("get")
@click.argument("container_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def get_container(container_id: str, api_key: str | None, backend_url: str | None) -> None:
    item = _client(api_key, backend_url).get(container_id)
    _echo(item.model_dump())


@containers.command("create")
@click.option("--name", required=True)
@click.option("--task-type", required=True)
@click.option("--definition", default="{}")
@click.option("--environment-config", default=None)
@click.option("--internal-url", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def create_container(
    name: str,
    task_type: str,
    definition: str,
    environment_config: str | None,
    internal_url: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    from synth_ai.sdk.containers import ContainerSpec

    spec = ContainerSpec(
        name=name,
        task_type=task_type,
        definition=_load_json_argument(definition),
        environment_config=_load_json_argument(environment_config) if environment_config else None,
        internal_url=internal_url,
    )
    item = _client(api_key, backend_url).create(spec)
    _echo(item.model_dump())


@containers.command("delete")
@click.argument("container_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def delete_container(container_id: str, api_key: str | None, backend_url: str | None) -> None:
    _client(api_key, backend_url).delete(container_id)
    _echo({"status": "deleted", "container_id": container_id})
