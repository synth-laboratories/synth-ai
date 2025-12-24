"""Status runs subcommand."""

from __future__ import annotations

import asyncio

import click

from .config import StatusConfig
from .utils import StatusAPIClient, print_json


@click.group("runs")
def runs_group() -> None:
    return None


@runs_group.command("list")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def list_runs(job_id: str, as_json: bool) -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            data = await client.list_job_runs(job_id)  # type: ignore[attr-defined]
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


__all__ = ["runs_group"]
