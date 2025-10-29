"""`synth models` command group."""

from __future__ import annotations

import asyncio

import click
from rich.json import JSON

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import console, models_table, print_json
from ..utils import bail, common_options, resolve_context_config


@click.group("models", help="Inspect fine-tuned models.")
@click.pass_context
def models_group(ctx: click.Context) -> None:  # pragma: no cover - Click wiring
    ctx.ensure_object(dict)


@models_group.command("list")
@common_options()
@click.option("--limit", type=int, default=None, help="Maximum number of models to return.")
@click.option("--type", "model_type", type=click.Choice(["rl", "sft"]), default=None, help="Filter by model type.")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def list_models(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    limit: int | None,
    model_type: str | None,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                models = await client.list_models(limit=limit, model_type=model_type)
                if output_json:
                    print_json(models)
                else:
                    console.print(models_table(models))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())


@models_group.command("get")
@common_options()
@click.argument("model_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def get_model(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    model_id: str,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        try:
            async with StatusAPIClient(cfg) as client:
                model = await client.get_model(model_id)
                if output_json:
                    print_json(model)
                else:
                    console.print(JSON.from_data(model))
        except StatusAPIError as exc:
            bail(f"Backend error: {exc}")

    asyncio.run(_run())
