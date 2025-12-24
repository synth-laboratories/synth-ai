"""Status files subcommand."""

from __future__ import annotations

import asyncio

import click

from .config import StatusConfig
from .utils import StatusAPIClient, print_json


@click.group("files")
def files_group() -> None:
    return None


@files_group.command("get")
@click.argument("file_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def get_file(file_id: str, as_json: bool) -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            data = await client.get_file(file_id)  # type: ignore[attr-defined]
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


__all__ = ["files_group"]
