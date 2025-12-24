"""Status summary command."""

from __future__ import annotations

import asyncio

import click

from synth_ai.cli.commands.status.errors import StatusAPIError
from .config import StatusConfig
from .utils import StatusAPIClient


@click.command("summary")
def summary_command() -> None:
    async def _run() -> None:
        config = StatusConfig()
        async with StatusAPIClient(config) as client:
            try:
                jobs = await client.list_jobs()  # type: ignore[attr-defined]
            except StatusAPIError:
                jobs = []
            try:
                await client.list_models()  # type: ignore[attr-defined]
            except StatusAPIError:
                pass
            try:
                await client.list_files()  # type: ignore[attr-defined]
            except StatusAPIError:
                pass

        click.echo("Training Jobs")
        for job in jobs:
            click.echo(str(job))

    asyncio.run(_run())


__all__ = ["summary_command"]
