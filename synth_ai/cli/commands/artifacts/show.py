"""Show artifact details command."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .client import ArtifactsClient
from .config import DEFAULT_TIMEOUT, resolve_backend_config
from .parsing import (
    detect_artifact_type,
    parse_model_id,
    parse_prompt_id,
)

console = Console()


def _format_model_details(data: dict[str, Any]) -> None:
    """Format model details."""
    model_id = data.get("id", "")
    model_type = data.get("type", "")
    base_model = data.get("base_model", "")
    job_id = data.get("job_id", "")
    status = data.get("status", "")
    created = data.get("created_at", "")
    
    lines = [
        f"[bold]Model:[/bold] {model_id}",
        f"[bold]Type:[/bold] {model_type}",
        f"[bold]Base Model:[/bold] {base_model}",
        f"[bold]Job ID:[/bold] {job_id}",
        f"[bold]Status:[/bold] {status}",
        f"[bold]Created:[/bold] {created}",
    ]
    
    if model_type == "rl":
        dtype = data.get("dtype", "")
        weights_path = data.get("weights_path", "")
        lines.extend([
            f"[bold]Dtype:[/bold] {dtype}",
            f"[bold]Weights Path:[/bold] {weights_path}",
        ])
    
    console.print(Panel("\n".join(lines), title="Model Details", border_style="blue"))


def _extract_prompt_messages(snapshot: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Extract prompt messages from snapshot payload.
    
    Handles multiple snapshot structures:
    1. Direct 'messages' array
    2. 'object' -> 'messages' array
    3. 'object' -> 'text_replacements' array (GEPA structure)
    4. 'initial_prompt' -> 'data' -> 'messages'
    """
    if not snapshot or not isinstance(snapshot, dict):
        return None
    
    # Structure 1: Direct messages
    if "messages" in snapshot:
        msgs = snapshot["messages"]
        if isinstance(msgs, list) and msgs:
            return msgs
    
    # Structure 2: object -> messages
    obj = snapshot.get("object", {})
    if isinstance(obj, dict):
        if "messages" in obj:
            msgs = obj["messages"]
            if isinstance(msgs, list) and msgs:
                return msgs
        
        # Structure 3: object -> text_replacements (GEPA)
        text_replacements = obj.get("text_replacements", [])
        if isinstance(text_replacements, list) and text_replacements:
            messages = []
            for replacement in text_replacements:
                if not isinstance(replacement, dict):
                    continue
                role = replacement.get("apply_to_role", "system")
                content = replacement.get("new_text", "")
                if content:
                    messages.append({"role": role, "content": content})
            if messages:
                return messages
    
    # Structure 4: initial_prompt -> data -> messages
    initial_prompt = snapshot.get("initial_prompt", {})
    if isinstance(initial_prompt, dict):
        data = initial_prompt.get("data", {})
        if isinstance(data, dict):
            msgs = data.get("messages", [])
            if isinstance(msgs, list) and msgs:
                return msgs
    
    return None


def _format_best_prompt(snapshot: dict[str, Any] | None, snapshot_id: str | None) -> None:
    """Format and display the best prompt."""
    messages = _extract_prompt_messages(snapshot)
    
    if not messages:
        console.print("[yellow]No prompt found in snapshot.[/yellow]")
        if snapshot:
            console.print(f"[dim]Snapshot structure: {list(snapshot.keys())[:10]}[/dim]")
        return
    
    console.print("\n[bold cyan]Best Optimized Prompt:[/bold cyan]")
    if snapshot_id:
        console.print(f"[dim]Snapshot ID: {snapshot_id}[/dim]\n")
    
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        pattern = msg.get("pattern", "")
        
        # Use content or pattern, whichever is available
        text = content or pattern
        
        if text:
            role_color = {
                "system": "blue",
                "user": "green",
                "assistant": "yellow",
            }.get(role, "white")
            
            console.print(f"[bold {role_color}]{role.upper()}:[/bold {role_color}]")
            console.print(Syntax(text, "text", theme="monokai", word_wrap=True))
            if i < len(messages):
                console.print()  # Add spacing between messages


def _format_prompt_details(data: dict[str, Any], verbose: bool = False) -> None:
    """Format prompt job details."""
    job_id = data.get("job_id", "")
    algorithm = data.get("algorithm", "")
    status = data.get("status", "")
    best_score = data.get("best_score")
    best_val_score = data.get("best_validation_score")
    created = data.get("created_at", "")
    finished = data.get("finished_at", "")
    best_snapshot_id = data.get("best_snapshot_id")
    best_snapshot = data.get("best_snapshot")
    
    # Summary panel
    lines = [
        f"[bold]Job ID:[/bold] {job_id}",
        f"[bold]Algorithm:[/bold] {algorithm or 'N/A'}",
        f"[bold]Status:[/bold] {status}",
    ]
    
    if best_score is not None:
        lines.append(f"[bold]Best Score:[/bold] {best_score:.3f}")
    else:
        lines.append("[bold]Best Score:[/bold] N/A")
    
    if best_val_score is not None:
        lines.append(f"[bold]Best Validation Score:[/bold] {best_val_score:.3f}")
    else:
        lines.append("[bold]Best Validation Score:[/bold] N/A")
    
    lines.extend([
        f"[bold]Created:[/bold] {created}",
        f"[bold]Finished:[/bold] {finished}" if finished else "[bold]Finished:[/bold] N/A",
    ])
    
    if best_snapshot_id:
        lines.append(f"[bold]Best Snapshot ID:[/bold] {best_snapshot_id}")
    
    console.print(Panel("\n".join(lines), title="Prompt Optimization Job", border_style="green"))
    
    # Show best prompt by default
    if best_snapshot:
        _format_best_prompt(best_snapshot, best_snapshot_id)
    
    # Show verbose details if requested
    if verbose:
        console.print("\n[bold cyan]Full Details:[/bold cyan]")
        
        metadata = data.get("metadata", {})
        if metadata:
            console.print("\n[bold]Metadata:[/bold]")
            # Show key metadata fields
            important_keys = [
                "algorithm", "prompt_best_score", "prompt_best_validation_score",
                "prompt_best_snapshot_id", "task_app_url", "config_source"
            ]
            for key in important_keys:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, dict | list):
                        console.print(f"  [bold]{key}:[/bold]")
                        console.print(Syntax(json.dumps(value, indent=2), "json"))
                    else:
                        console.print(f"  [bold]{key}:[/bold] {value}")
            
            # Show other metadata keys
            other_keys = [k for k in metadata if k not in important_keys]
            if other_keys:
                console.print(f"\n  [dim]Other metadata keys: {', '.join(other_keys[:10])}[/dim]")
        
        # Show full snapshot if available
        if best_snapshot:
            console.print("\n[bold]Full Best Snapshot:[/bold]")
            console.print(Syntax(json.dumps(best_snapshot, indent=2, default=str), "json"))


@click.command("show", help="Show detailed information about an artifact.")
@click.argument("artifact_id", required=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format. Use 'json' to export all data as JSON.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show verbose details (full metadata, snapshot, etc.). Only applies to prompts.",
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
def show_command(
    artifact_id: str,
    output_format: str,
    verbose: bool,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """Show detailed information about a model or prompt artifact.
    
    For prompts, by default shows:
    - Job summary (algorithm, scores, status)
    - Best optimized prompt (extracted from snapshot)
    
    Use --verbose to see full metadata and snapshot details.
    Use --format json to export all data as JSON.
    """
    config = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
    
    async def _run() -> None:
        client = ArtifactsClient(config.api_base_url, config.api_key, timeout=config.timeout)
        try:
            # Determine artifact type using centralized parsing
            artifact_type = detect_artifact_type(artifact_id)
            
            if artifact_type == "model":
                # Validate and parse model ID
                try:
                    parsed = parse_model_id(artifact_id)
                except ValueError as e:
                    raise click.ClickException(f"Invalid model ID format: {e}") from e
                
                data = await client.get_model(artifact_id)
                if output_format == "json":
                    console.print(json.dumps(data, indent=2, default=str))
                else:
                    _format_model_details(data)
            
            elif artifact_type == "prompt":
                # Validate and parse prompt ID
                try:
                    parsed = parse_prompt_id(artifact_id)
                except ValueError as e:
                    raise click.ClickException(f"Invalid prompt ID format: {e}") from e
                
                data = await client.get_prompt(parsed.job_id)
                if output_format == "json":
                    # JSON format - output everything
                    console.print(json.dumps(data, indent=2, default=str))
                else:
                    # Table format - show summary + best prompt, optionally verbose
                    _format_prompt_details(data, verbose=verbose)
            
            else:
                # Unknown type - try both endpoints
                try:
                    data = await client.get_model(artifact_id)
                    if output_format == "json":
                        console.print(json.dumps(data, indent=2, default=str))
                    else:
                        _format_model_details(data)
                    return
                except Exception:
                    pass
                
                # Try as prompt
                try:
                    parsed = parse_prompt_id(artifact_id)
                    data = await client.get_prompt(parsed.job_id)
                    if output_format == "json":
                        console.print(json.dumps(data, indent=2, default=str))
                    else:
                        _format_prompt_details(data, verbose=verbose)
                except Exception as e:
                    raise click.ClickException(
                        f"Could not identify artifact type. Tried as model and prompt, but both failed. "
                        f"Last error: {e}"
                    ) from e
        except click.ClickException:
            raise
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.ClickException(str(e)) from e
    
    asyncio.run(_run())

