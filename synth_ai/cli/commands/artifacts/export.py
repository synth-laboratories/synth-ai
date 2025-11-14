"""Export model to HuggingFace command."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from .client import ArtifactsClient
from .config import DEFAULT_TIMEOUT, resolve_backend_config
from .parsing import (
    parse_model_id,
    resolve_wasabi_key_for_model,
    validate_model_id,
)

console = Console()


async def _resolve_model_storage_path(
    client: ArtifactsClient,
    model_id: str,
    prefer_merged: bool,
) -> tuple[str, str]:
    """Resolve model ID to Wasabi storage path and artifact kind."""
    # Get model details
    model_data = await client.get_model(model_id)
    model_type = model_data.get("type")
    base_model = model_data.get("base_model", "")
    job_id = model_data.get("job_id", "")
    
    if model_type == "fine_tuned":
        # Fine-tuned model: check Wasabi availability first
        try:
            wasabi_data = await client.get_models_on_wasabi()
            models = wasabi_data.get("models", [])
            # Find matching model
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
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check Wasabi availability: {e}[/yellow]")
        
        # Fallback: construct path using parsing utilities
        try:
            parsed = parse_model_id(model_id)
            wasabi_key = resolve_wasabi_key_for_model(parsed, prefer_merged=prefer_merged)
            artifact_kind = "training_file" if prefer_merged else "lora"
            return wasabi_key, artifact_kind
        except Exception:
            # Last resort fallback
            safe_base = base_model.replace("/", "-").replace(":", "-")
            if prefer_merged:
                wasabi_key = f"models/{safe_base}-{job_id}-fp16.tar.gz"
                artifact_kind = "training_file"
            else:
                wasabi_key = f"models/{base_model}/ft-{job_id}/adapter.tar.gz"
                artifact_kind = "lora"
            return wasabi_key, artifact_kind
    elif model_type == "rl":
        # RL model: use weights_path
        weights_path = model_data.get("weights_path", "")
        if not weights_path:
            raise click.ClickException(f"RL model {model_id} has no weights_path")
        # Determine artifact kind from path
        artifact_kind = "lora" if "adapter" in weights_path.lower() else "training_file"
        return weights_path, artifact_kind
    else:
        raise click.ClickException(f"Unknown model type: {model_type}")


@click.command("export", help="Export a model to HuggingFace Hub.")
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
def export_command(
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
            # Validate and parse model ID
            if not validate_model_id(model_id):
                raise click.ClickException(
                    f"Invalid model ID format: {model_id}. "
                    f"Expected format: peft:BASE_MODEL:JOB_ID, ft:BASE_MODEL:JOB_ID, or rl:BASE_MODEL:JOB_ID"
                )
            
            # Validate model ID format (parsed is used for validation)
            parse_model_id(model_id)
            
            # Resolve storage path
            console.print(f"[yellow]Resolving storage path for {model_id}...[/yellow]")
            wasabi_key, artifact_kind = await _resolve_model_storage_path(
                client, model_id, prefer_merged
            )
            
            # Get model details for base_model
            model_data = await client.get_model(model_id)
            base_model = model_data.get("base_model")
            
            # Parse tags
            tag_list = [t.strip() for t in tags.split(",")] if tags else None
            
            # Export to HuggingFace
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
            
            # Show result
            repo_url = result.get("repo_url") or f"https://huggingface.co/{repo_id}"
            console.print(f"[green]✓ Successfully exported to {repo_url}[/green]")
            if isinstance(result, dict):
                console.print(f"Repository: {repo_id}")
                console.print(f"Visibility: {'private' if private else 'public'}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.ClickException(str(e)) from e
    
    asyncio.run(_run())

