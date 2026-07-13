"""Pools CLI commands."""

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
    from synth_ai.sdk.pools import ContainerPoolsClient

    return ContainerPoolsClient(api_key=api_key, backend_base=backend_url)


def _echo(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.group()
def pools() -> None:
    """Manage container pools and rollouts."""


@pools.command("list")
@click.option("--state", default=None)
@click.option("--limit", default=100, type=int)
@click.option("--cursor", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def list_pools(
    state: str | None,
    limit: int,
    cursor: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).list(state=state, limit=limit, cursor=cursor))


@pools.command("get")
@click.argument("pool_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def get_pool(pool_id: str, api_key: str | None, backend_url: str | None) -> None:
    _echo(_client(api_key, backend_url).get(pool_id))


@pools.command("create")
@click.option("--request", "request_json", required=True, help="JSON object or @path/to/file.json")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def create_pool(request_json: str, api_key: str | None, backend_url: str | None) -> None:
    _echo(_client(api_key, backend_url).create(_load_json_argument(request_json)))


@pools.command("update")
@click.argument("pool_id")
@click.option("--request", "request_json", required=True, help="JSON object or @path/to/file.json")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def update_pool(
    pool_id: str,
    request_json: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).update(pool_id, _load_json_argument(request_json)))


@pools.command("delete")
@click.argument("pool_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def delete_pool(pool_id: str, api_key: str | None, backend_url: str | None) -> None:
    _echo(_client(api_key, backend_url).delete(pool_id))


@pools.command("rollout-create")
@click.argument("pool_id")
@click.option("--request", "request_json", required=True, help="JSON object or @path/to/file.json")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def rollout_create(
    pool_id: str,
    request_json: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).rollouts.create(pool_id, _load_json_argument(request_json)))


@pools.command("rollout-get")
@click.argument("pool_id")
@click.argument("rollout_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def rollout_get(
    pool_id: str,
    rollout_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).rollouts.get(pool_id, rollout_id))


@pools.command("rollout-summary")
@click.argument("pool_id")
@click.argument("rollout_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def rollout_summary(
    pool_id: str,
    rollout_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).rollouts.summary(pool_id, rollout_id))


@pools.command("rollout-artifacts")
@click.argument("pool_id")
@click.argument("rollout_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def rollout_artifacts(
    pool_id: str,
    rollout_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(_client(api_key, backend_url).rollouts.artifacts(pool_id, rollout_id))
