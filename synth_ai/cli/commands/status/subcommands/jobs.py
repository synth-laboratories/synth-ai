"""`synth jobs` command group implementation."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import (
    console,
    events_panel,
    job_panel,
    jobs_table,
    metrics_table,
    print_json,
    runs_table,
)
from ..utils import bail, common_options, parse_relative_time, resolve_context_config


@click.group("jobs", help="Manage training jobs.")
@click.pass_context
def jobs_group(ctx: click.Context) -> None:  # pragma: no cover - Click wiring
    ctx.ensure_object(dict)


def _print_or_json(items: Any, output_json: bool) -> None:
    if output_json:
        print_json(items)
    elif isinstance(items, list):
        console.print(jobs_table(items))
    else:
        console.print(job_panel(items))


@jobs_group.command("list")
@common_options()
@click.option(
    "--status",
    type=click.Choice(["queued", "running", "succeeded", "failed", "cancelled"]),
    help="Filter by job status.",
)
@click.option(
    "--type",
    "job_type",
    type=click.Choice(["sft_offline", "sft_online", "rl_online", "dpo", "sft"]),
    help="Filter by training job type.",
)
@click.option("--created-after", help="Filter by creation date (ISO8601 or relative like '24h').")
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--json", "output_json", is_flag=True, help="Emit raw JSON.")
@click.pass_context
def list_jobs(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    status: str | None,
    job_type: str | None,
    created_after: str | None,
    limit: int,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)
    created_filter = parse_relative_time(created_after)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                jobs = await client.list_jobs(
                    status=status,
                    job_type=job_type,
                    created_after=created_filter,
                    limit=limit,
                )
                _print_or_json(jobs, output_json)
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("get")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def get_job(
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
                job = await client.get_job(job_id)
                _print_or_json(job, output_json)
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("history")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_history(
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


@jobs_group.command("timeline")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_timeline(
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
                timeline = await client.get_job_timeline(job_id)
                if output_json:
                    print_json(timeline)
                else:
                    console.print(events_panel(timeline))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("metrics")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_metrics(
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
                metrics = await client.get_job_metrics(job_id)
                if output_json:
                    print_json(metrics)
                else:
                    console.print(metrics_table(metrics))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("config")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_config(
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
                config = await client.get_job_config(job_id)
                if output_json:
                    print_json(config)
                else:
                    console.print(job_panel({"job_id": job_id, "config": config}))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("status")
@common_options()
@click.argument("job_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_status(
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
                status = await client.get_job_status(job_id)
                if output_json:
                    print_json(status)
                else:
                    console.print(f"[bold]{job_id}[/bold]: {status.get('status', 'unknown')}")
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("cancel")
@common_options()
@click.argument("job_id")
@click.pass_context
def cancel_job(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    job_id: str,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                resp = await client.cancel_job(job_id)
                console.print(resp.get("message") or f"[yellow]Cancellation requested for {job_id}[/yellow]")
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@jobs_group.command("logs")
@common_options()
@click.argument("job_id")
@click.option("--since", help="Only show events emitted after the provided timestamp/relative offset.")
@click.option("--tail", type=int, help="Show only the last N events.")
@click.option("--follow/--no-follow", default=False, help="Poll for new events.")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def job_logs(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    job_id: str,
    since: str | None,
    tail: int | None,
    follow: bool,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)
    since_filter = parse_relative_time(since)

    async def _loop() -> None:
        seen_ids: set[str] = set()
        cursor: str | None = None
        try:
            async with StatusAPIClient(cfg) as client:
                while True:
                    events = await client.get_job_events(
                        job_id,
                        since=cursor or since_filter,
                        limit=tail,
                        after=cursor,
                    )
                    new_events: list[dict[str, Any]] = []
                    for event in events:
                        event_id = str(event.get("event_id") or event.get("id") or event.get("timestamp"))
                        if event_id in seen_ids:
                            continue
                        seen_ids.add(event_id)
                        new_events.append(event)
                    if new_events:
                        cursor = str(new_events[-1].get("event_id") or new_events[-1].get("id") or "")
                        if output_json:
                            print_json(new_events)
                        else:
                            console.print(events_panel(new_events))
                    if not follow:
                        break
                    await asyncio.sleep(2.0)
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_loop())
