"""Download optimized prompts command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore
from rich.console import Console

from .client import ArtifactsClient
from .config import DEFAULT_TIMEOUT, resolve_backend_config

console = Console()


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    """Extract prompt text from snapshot payload."""
    if isinstance(payload, dict):
        # Try to find messages or text fields
        if "messages" in payload:
            messages = payload["messages"]
            if isinstance(messages, list):
                texts = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        texts.append(msg["content"])
                return "\n\n".join(texts)
        if "text" in payload:
            return str(payload["text"])
        if "prompt" in payload:
            return str(payload["prompt"])
    return json.dumps(payload, indent=2)


@click.command("download", help="Download optimized prompts.")
@click.argument("job_id", required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file or directory path.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "text"]),
    default="json",
    help="Output format.",
)
@click.option(
    "--snapshot-id",
    help="Download specific snapshot (default: best snapshot).",
)
@click.option(
    "--all-snapshots",
    is_flag=True,
    help="Download all snapshots from the job.",
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
def download_command(
    job_id: str,
    output: str | None,
    output_format: str,
    snapshot_id: str | None,
    all_snapshots: bool,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """Download optimized prompts from a prompt optimization job."""
    config = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
    
    async def _run() -> None:
        client = ArtifactsClient(config.api_base_url, config.api_key, timeout=config.timeout)
        try:
            # Get job details
            job_data = await client.get_prompt(job_id)
            best_snapshot_id = job_data.get("best_snapshot_id")
            
            if all_snapshots:
                # Download all snapshots
                artifacts = await client.list_prompt_snapshots(job_id)
                if not artifacts:
                    console.print(f"[yellow]No snapshots found for job {job_id}[/yellow]")
                    return
                
                # Determine output directory
                if output:
                    output_path = Path(output)
                    output_dir = output_path.parent if output_path.suffix else output_path
                else:
                    output_dir = Path(f"./prompts/{job_id}")
                
                output_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[yellow]Downloading {len(artifacts)} snapshots to {output_dir}...[/yellow]")
                
                for artifact in artifacts:
                    snap_id = artifact.get("snapshot_id")
                    if not snap_id:
                        continue
                    
                    snapshot_data = await client.get_prompt_snapshot(job_id, snap_id)
                    payload = snapshot_data.get("payload", {})
                    
                    # Determine filename
                    if output_format == "text":
                        content = _extract_prompt_text(payload)
                        ext = "txt"
                    elif output_format == "yaml":
                        if yaml is None:
                            raise click.ClickException("yaml format requires PyYAML. Install with: pip install pyyaml")
                        content = yaml.dump(payload, default_flow_style=False)
                        ext = "yaml"
                    else:
                        content = json.dumps(payload, indent=2, default=str)
                        ext = "json"
                    
                    file_path = output_dir / f"snapshot_{snap_id}.{ext}"
                    file_path.write_text(content)
                    console.print(f"  [green]✓[/green] {file_path}")
            else:
                # Download single snapshot (best or specified)
                target_snapshot_id = snapshot_id or best_snapshot_id
                if not target_snapshot_id:
                    raise click.ClickException(f"No snapshot found for job {job_id}")
                
                snapshot_data = await client.get_prompt_snapshot(job_id, target_snapshot_id)
                payload = snapshot_data.get("payload", {})
                
                # Determine output path
                if output:
                    output_path = Path(output)
                else:
                    if output_format == "text":
                        ext = "txt"
                    elif output_format == "yaml":
                        ext = "yaml"
                    else:
                        ext = "json"
                    output_path = Path(f"./prompts/{job_id}/best.{ext}")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content
                if output_format == "text":
                    content = _extract_prompt_text(payload)
                elif output_format == "yaml":
                    if yaml is None:
                        raise click.ClickException("yaml format requires PyYAML. Install with: pip install pyyaml")
                    content = yaml.dump(payload, default_flow_style=False)
                else:
                    content = json.dumps(payload, indent=2, default=str)
                
                output_path.write_text(content)
                console.print(f"[green]✓ Downloaded to {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.ClickException(str(e)) from e
    
    asyncio.run(_run())

