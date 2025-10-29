"""`synth status summary` command."""

from __future__ import annotations

import asyncio

import click

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import console, files_table, jobs_table, models_table
from ..utils import common_options, resolve_context_config


@click.command("summary", help="Show a condensed overview of recent jobs, models, and files.")
@common_options()
@click.option("--limit", default=5, show_default=True, type=int, help="Rows per section.")
@click.pass_context
def summary_command(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    limit: int,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        async with StatusAPIClient(cfg) as client:
            try:
                jobs = await client.list_jobs(limit=limit)
            except StatusAPIError:
                jobs = []
            try:
                models = await client.list_models(limit=limit)
            except StatusAPIError:
                models = []
            try:
                files = await client.list_files(limit=limit)
            except StatusAPIError:
                files = []
        return jobs, models, files

    jobs, models, files = asyncio.run(_run())
    console.print(jobs_table(jobs[:limit]))
    console.print(models_table(models[:limit]))
    console.print(files_table(files[:limit]))
