"""Status jobs subcommand."""

from __future__ import annotations

import asyncio

import click

from .config import StatusConfig
from .utils import StatusAPIClient, print_json


@click.group("jobs")
def jobs_group() -> None:
    return None


@jobs_group.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--status", default=None)
def list_jobs(as_json: bool, status: str | None) -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            data = await client.list_jobs(status=status)  # type: ignore[attr-defined]
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@jobs_group.command("logs")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--tail", default=None, type=int)
def logs(job_id: str, as_json: bool, tail: int | None) -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            data = await client.get_job_events(job_id, limit=tail)  # type: ignore[attr-defined]
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


__all__ = ["jobs_group"]
