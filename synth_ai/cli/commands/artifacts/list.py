"""List artifacts command."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from .client import ArtifactsClient
from .config import DEFAULT_TIMEOUT, resolve_backend_config

console = Console()


def _format_table(data: dict[str, Any]) -> None:
    """Format artifacts as a table."""
    # Fine-tuned models
    ft_models = data.get("fine_tuned_models", [])
    if ft_models:
        console.print(f"\n[bold]Fine-tuned Models ({len(ft_models)})[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Model ID")
        table.add_column("Base Model")
        table.add_column("Created")
        table.add_column("Job ID")
        for model in ft_models:
            model_id = model.get("id", "")
            base = model.get("base_model", "")
            created = model.get("created_at", "")
            job_id = model.get("job_id", "")
            table.add_row(model_id[:40], base[:30], str(created)[:19], job_id[:20])
        console.print(table)
    
    # RL models
    rl_models = data.get("rl_models", [])
    if rl_models:
        console.print(f"\n[bold]RL Models ({len(rl_models)})[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Model ID")
        table.add_column("Base Model")
        table.add_column("Created")
        table.add_column("Job ID")
        for model in rl_models:
            model_id = model.get("id", "")
            base = model.get("base_model", "")
            created = model.get("created_at", "")
            job_id = model.get("job_id", "")
            table.add_row(model_id[:40], base[:30], str(created)[:19], job_id[:20])
        console.print(table)
    
    # Prompts
    prompts = data.get("prompts", [])
    if prompts:
        console.print(f"\n[bold]Optimized Prompts ({len(prompts)})[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Job ID")
        table.add_column("Algorithm")
        table.add_column("Best Score")
        table.add_column("Created")
        for prompt in prompts:
            job_id = prompt.get("job_id", "")
            algorithm = prompt.get("algorithm", "")
            score = prompt.get("best_score")
            created = prompt.get("created_at", "")
            score_str = f"{score:.3f}" if score is not None else "N/A"
            table.add_row(job_id[:30], algorithm[:10], score_str, str(created)[:19])
        console.print(table)
    
    # Summary
    summary = data.get("summary", {})
    total = summary.get("total_count", 0)
    if total == 0:
        console.print("\n[yellow]No artifacts found.[/yellow]")


@click.command("list", help="List all artifacts (models and prompts).")
@click.option(
    "--type",
    "artifact_type",
    type=click.Choice(["models", "prompts", "all"]),
    default="all",
    help="Filter by artifact type.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--limit",
    default=50,
    type=int,
    show_default=True,
    help="Maximum items per type.",
)
@click.option(
    "--status",
    type=click.Choice(["succeeded", "failed", "running"]),
    default="succeeded",
    help="Filter by status.",
)
@click.option(
    "--base-url",
    envvar="BACKEND_BASE_URL",
    default=None,
    help="Backend base URL.",
)
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    default=None,
    help="API key.",
)
@click.option(
    "--timeout",
    default=DEFAULT_TIMEOUT,
    type=float,
    show_default=True,
    help="Request timeout in seconds.",
)
def list_command(
    artifact_type: str,
    output_format: str,
    limit: int,
    status: str,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """List all artifacts (fine-tuned models, RL models, and optimized prompts)."""
    config = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
    
    async def _run() -> None:
        client = ArtifactsClient(config.api_base_url, config.api_key, timeout=config.timeout)
        try:
            data = await client.list_artifacts(
                artifact_type=artifact_type if artifact_type != "all" else None,
                status=status,
                limit=limit,
            )
            if output_format == "json":
                console.print(json.dumps(data, indent=2, default=str))
            else:
                _format_table(data)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.ClickException(str(e)) from e
    
    asyncio.run(_run())

