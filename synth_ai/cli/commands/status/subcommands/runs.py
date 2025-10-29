"""`synth runs` command group."""

from __future__ import annotations

import asyncio

import click

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import console, events_panel, print_json, runs_table
from ..utils import bail, common_options, parse_relative_time, resolve_context_config


@click.group("runs", help="Inspect individual job runs/attempts.")
@click.pass_context
def runs_group(ctx: click.Context) -> None:  # pragma: no cover - Click wiring
    ctx.ensure_object(dict)


@runs_group.command("list")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def list_runs(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    job_id: str,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                runs = await client.list_job_runs(job_id)
                if output_json:
                    print_json(runs)
                else:
                    console.print(runs_table(runs))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@runs_group.command("logs")
@common_options()
@click.argument("job_id")
@click.option("--run", "run_id", required=True, help="Run identifier (number or ID) to inspect.")
@click.option("--since", help="Filter events after the supplied timestamp/relative offset.")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def run_logs(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    job_id: str,
    run_id: str,
    since: str | None,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)
    since_filter = parse_relative_time(since)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                events = await client.get_job_events(job_id, since=since_filter, run_id=run_id)
                if output_json:
                    print_json(events)
                else:
                    console.print(events_panel(events))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())
