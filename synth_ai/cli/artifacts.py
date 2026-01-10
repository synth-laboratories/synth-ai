"""Artifacts command."""

import asyncio
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from synth_ai.sdk.artifacts.client import ArtifactsClient
from synth_ai.sdk.artifacts.config import DEFAULT_TIMEOUT, resolve_backend_config
from synth_ai.sdk.artifacts.parsing import (
    detect_artifact_type,
    parse_model_id,
    parse_prompt_id,
    resolve_wasabi_key_for_model,
    validate_model_id,
)

console = Console()


def _format_table(data: dict[str, Any]) -> None:
    """Format artifacts as a table."""
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

    summary = data.get("summary", {})
    total = summary.get("total_count", 0)
    if total == 0:
        console.print("\n[yellow]No artifacts found.[/yellow]")


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    """Extract prompt text from snapshot payload."""
    if isinstance(payload, dict):
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


async def _resolve_model_storage_path(
    client: ArtifactsClient,
    model_id: str,
    prefer_merged: bool,
) -> tuple[str, str]:
    """Resolve model ID to Wasabi storage path and artifact kind."""
    model_data = await client.get_model(model_id)
    model_type = model_data.get("type")
    base_model = model_data.get("base_model", "")
    job_id = model_data.get("job_id", "")

    if model_type == "fine_tuned":
        try:
            wasabi_data = await client.get_models_on_wasabi()
            models = wasabi_data.get("models", [])
            for model_info in models:
                if model_info.get("id") == model_id:
                    wasabi_info = model_info.get("wasabi", {})
                    merged_info = wasabi_info.get("merged", {})
                    adapter_info = wasabi_info.get("adapter", {})

                    if prefer_merged and merged_info.get("present"):
                        wasabi_key = merged_info.get("key", "")
                        if wasabi_key:
                            return wasabi_key, "training_file"
                    elif adapter_info.get("present"):
                        wasabi_key = adapter_info.get("key", "")
                        if wasabi_key:
                            return wasabi_key, "lora"
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not check Wasabi availability: {exc}[/yellow]")

        try:
            parsed = parse_model_id(model_id)
            wasabi_key = resolve_wasabi_key_for_model(parsed, prefer_merged=prefer_merged)
            artifact_kind = "training_file" if prefer_merged else "lora"
            return wasabi_key, artifact_kind
        except Exception:
            safe_base = base_model.replace("/", "-").replace(":", "-")
            if prefer_merged:
                wasabi_key = f"models/{safe_base}-{job_id}-fp16.tar.gz"
                artifact_kind = "training_file"
            else:
                wasabi_key = f"models/{base_model}/ft-{job_id}/adapter.tar.gz"
                artifact_kind = "lora"
            return wasabi_key, artifact_kind

    if model_type == "rl":
        weights_path = model_data.get("weights_path", "")
        if not weights_path:
            raise click.ClickException(f"RL model {model_id} has no weights_path")
        artifact_kind = "lora" if "adapter" in weights_path.lower() else "training_file"
        return weights_path, artifact_kind

    raise click.ClickException(f"Unknown model type: {model_type}")


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
        lines.extend(
            [
                f"[bold]Dtype:[/bold] {dtype}",
                f"[bold]Weights Path:[/bold] {weights_path}",
            ]
        )

    console.print(Panel("\n".join(lines), title="Model Details", border_style="blue"))


def _extract_prompt_messages(snapshot: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Extract prompt messages from snapshot payload."""
    if not snapshot or not isinstance(snapshot, dict):
        return None

    if "messages" in snapshot:
        msgs = snapshot["messages"]
        if isinstance(msgs, list) and msgs:
            return msgs

    obj = snapshot.get("object", {})
    if isinstance(obj, dict):
        if "messages" in obj:
            msgs = obj["messages"]
            if isinstance(msgs, list) and msgs:
                return msgs

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
                console.print()


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

    lines.extend(
        [
            f"[bold]Created:[/bold] {created}",
            f"[bold]Finished:[/bold] {finished}" if finished else "[bold]Finished:[/bold] N/A",
        ]
    )

    if best_snapshot_id:
        lines.append(f"[bold]Best Snapshot ID:[/bold] {best_snapshot_id}")

    console.print(Panel("\n".join(lines), title="Prompt Optimization Job", border_style="green"))

    _format_best_prompt(best_snapshot, best_snapshot_id)

    if verbose:
        console.print("\n[bold]Full Job Data:[/bold]")
        console.print(Syntax(json.dumps(data, indent=2, default=str), "json"))
        if best_snapshot:
            console.print("\n[bold]Full Best Snapshot:[/bold]")
            console.print(Syntax(json.dumps(best_snapshot, indent=2, default=str), "json"))


@click.group()
def artifacts() -> None:
    """Manage artifacts (models and optimized prompts)."""
    pass


@artifacts.command()
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
def list(
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
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise click.ClickException(str(exc)) from exc

    asyncio.run(_run())


@artifacts.command()
@click.argument("model_id", required=True)
@click.option(
    "--repo-id",
    required=True,
    help="HuggingFace repository ID (e.g., username/model-name).",
)
@click.option(
    "--private/--public",
    default=True,
    help="Repository visibility (default: private).",
)
@click.option(
    "--prefer-merged/--prefer-adapter",
    default=True,
    help="Prefer merged model over adapter (default: merged).",
)
@click.option(
    "--tags",
    help="Comma-separated tags for the repository.",
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
def export(
    model_id: str,
    repo_id: str,
    private: bool,
    prefer_merged: bool,
    tags: str | None,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """Export a fine-tuned or RL model to HuggingFace Hub."""
    config = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        client = ArtifactsClient(config.api_base_url, config.api_key, timeout=config.timeout)
        try:
            if not validate_model_id(model_id):
                raise click.ClickException(
                    f"Invalid model ID format: {model_id}. "
                    "Expected format: peft:BASE_MODEL:JOB_ID, ft:BASE_MODEL:JOB_ID, or rl:BASE_MODEL:JOB_ID"
                )

            parse_model_id(model_id)

            console.print(f"[yellow]Resolving storage path for {model_id}...[/yellow]")
            wasabi_key, artifact_kind = await _resolve_model_storage_path(
                client, model_id, prefer_merged
            )

            model_data = await client.get_model(model_id)
            base_model = model_data.get("base_model")

            tag_list = [t.strip() for t in tags.split(",")] if tags else None

            console.print(f"[yellow]Exporting to HuggingFace: {repo_id}...[/yellow]")
            result = await client.export_to_huggingface(
                wasabi_key=wasabi_key,
                repo_id=repo_id,
                repo_type="model",
                artifact_kind=artifact_kind,
                base_model=base_model,
                visibility="private" if private else "public",
                tags=tag_list,
            )

            repo_url = result.get("repo_url") or f"https://huggingface.co/{repo_id}"
            console.print(f"[green]✓ Successfully exported to {repo_url}[/green]")
            if isinstance(result, dict):
                console.print(f"Repository: {repo_id}")
                console.print(f"Visibility: {'private' if private else 'public'}")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise click.ClickException(str(exc)) from exc

    asyncio.run(_run())


@artifacts.command()
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
def download(
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
            job_data = await client.get_prompt(job_id)
            best_snapshot_id = job_data.get("best_snapshot_id")

            if all_snapshots:
                artifacts = await client.list_prompt_snapshots(job_id)
                if not artifacts:
                    console.print(f"[yellow]No snapshots found for job {job_id}[/yellow]")
                    return

                if output:
                    output_path = Path(output)
                    output_dir = output_path.parent if output_path.suffix else output_path
                else:
                    output_dir = Path(f"./prompts/{job_id}")

                output_dir.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"[yellow]Downloading {len(artifacts)} snapshots to {output_dir}...[/yellow]"
                )

                for artifact in artifacts:
                    snap_id = artifact.get("snapshot_id")
                    if not snap_id:
                        continue

                    snapshot_data = await client.get_prompt_snapshot(job_id, snap_id)
                    payload = snapshot_data.get("payload", {})

                    if output_format == "text":
                        content = _extract_prompt_text(payload)
                        ext = "txt"
                    elif output_format == "yaml":
                        if yaml is None:
                            raise click.ClickException(
                                "yaml format requires PyYAML. Install with: pip install pyyaml"
                            )
                        content = yaml.dump(payload, default_flow_style=False)
                        ext = "yaml"
                    else:
                        content = json.dumps(payload, indent=2, default=str)
                        ext = "json"

                    file_path = output_dir / f"snapshot_{snap_id}.{ext}"
                    file_path.write_text(content)
                    console.print(f"  [green]✓[/green] {file_path}")
            else:
                target_snapshot_id = snapshot_id or best_snapshot_id
                if not target_snapshot_id:
                    raise click.ClickException(f"No snapshot found for job {job_id}")

                snapshot_data = await client.get_prompt_snapshot(job_id, target_snapshot_id)
                payload = snapshot_data.get("payload", {})

                output_path = Path(output) if output else None

                if output_format == "text":
                    content = _extract_prompt_text(payload)
                    ext = "txt"
                elif output_format == "yaml":
                    if yaml is None:
                        raise click.ClickException(
                            "yaml format requires PyYAML. Install with: pip install pyyaml"
                        )
                    content = yaml.dump(payload, default_flow_style=False)
                    ext = "yaml"
                else:
                    content = json.dumps(payload, indent=2, default=str)
                    ext = "json"

                if output_path is None:
                    output_path = Path(f"./prompt_{job_id}_{target_snapshot_id}.{ext}")
                elif output_path.is_dir():
                    output_path = output_path / f"snapshot_{target_snapshot_id}.{ext}"

                output_path.write_text(content)
                console.print(f"[green]✓[/green] Saved to {output_path}")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise click.ClickException(str(exc)) from exc

    asyncio.run(_run())


@artifacts.command()
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
def show(
    artifact_id: str,
    output_format: str,
    verbose: bool,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """Show detailed information about a model or prompt artifact."""
    config = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)

    async def _run() -> None:
        client = ArtifactsClient(config.api_base_url, config.api_key, timeout=config.timeout)
        try:
            artifact_type = detect_artifact_type(artifact_id)

            if artifact_type == "model":
                model_id = parse_model_id(artifact_id)
                data = await client.get_model(model_id)
                if output_format == "json":
                    console.print(json.dumps(data, indent=2, default=str))
                else:
                    _format_model_details(data)
            elif artifact_type == "prompt":
                prompt_id = parse_prompt_id(artifact_id)
                data = await client.get_prompt(prompt_id)
                if output_format == "json":
                    console.print(json.dumps(data, indent=2, default=str))
                else:
                    _format_prompt_details(data, verbose=verbose)
            else:
                raise click.ClickException(
                    f"Invalid artifact ID format: {artifact_id}. "
                    "Expected model ID (peft:..., ft:..., rl:...) or prompt ID (prompt:...)"
                )
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise click.ClickException(str(exc)) from exc

    asyncio.run(_run())
