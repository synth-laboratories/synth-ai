"""`synth files` command group."""

from __future__ import annotations

import asyncio

import click
from rich.json import JSON

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import console, files_table, print_json
from ..utils import bail, common_options, resolve_context_config


@click.group("files", help="Manage training files.")
@click.pass_context
def files_group(ctx: click.Context) -> None:  # pragma: no cover - Click wiring
    ctx.ensure_object(dict)


@files_group.command("list")
@common_options()
@click.option("--purpose", type=click.Choice(["fine-tune", "validation"]))
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def list_files(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    purpose: str | None,
    limit: int,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                files = await client.list_files(purpose=purpose, limit=limit)
                if output_json:
                    print_json(files)
                else:
                    console.print(files_table(files))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@files_group.command("get")
@common_options()
@click.argument("file_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def get_file(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    file_id: str,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                file_info = await client.get_file(file_id)
                if output_json:
                    print_json(file_info)
                else:
                    console.print(JSON.from_data(file_info))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())
