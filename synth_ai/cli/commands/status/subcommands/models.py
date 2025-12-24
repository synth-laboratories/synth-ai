"""Status models subcommand."""

from __future__ import annotations

import asyncio

import click

from .config import StatusConfig
from .utils import StatusAPIClient, print_json


@click.group("models")
def models_group() -> None:
    return None


@models_group.command("list")
@click.option("--limit", default=None, type=int)
@click.option("--type", "model_type", default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def list_models(limit: int | None, model_type: str | None, as_json: bool) -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            data = await client.list_models(limit=limit, model_type=model_type)  # type: ignore[attr-defined]
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


__all__ = ["models_group"]
