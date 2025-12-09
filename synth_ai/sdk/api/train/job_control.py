"""CLI commands for controlling running GEPA jobs.

Provides pause, cancel, message-proposer, and other job control commands.
"""

from __future__ import annotations

import json
import os
from typing import Any

import click

from synth_ai.cli.lib.env import get_synth_and_env_keys, load_env_file
from synth_ai.sdk.api.train.configs.prompt_learning import (
    GEPACommand,
    GEPACommandResult,
    GEPACommandType,
)
from synth_ai.sdk.learning.prompt_learning_client import (
    cancel_job,
    checkpoint_now,
    extend_budget,
    extend_generations,
    message_proposer,
    pause_job,
    ping_job,
    update_config,
)

from .utils import ensure_api_base


def _default_backend() -> str:
    """Resolve backend URL with proper production default."""
    explicit = os.environ.get("BACKEND_BASE_URL", "").strip()
    if explicit:
        return explicit

    override = os.environ.get("BACKEND_OVERRIDE", "").strip()
    if override:
        return override

    # Default to production
    return "https://api.synth.run/api"


def _get_backend_and_key(
    backend_override: str | None, env_file: str | None
) -> tuple[str, str]:
    """Get backend URL and API key from environment or overrides."""
    load_env_file()

    if env_file:
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(env_file), override=True)

    synth_api_key, _ = get_synth_and_env_keys(env_file)

    if backend_override:
        backend_base = ensure_api_base(backend_override.strip())
    else:
        backend_base = ensure_api_base(_default_backend())

    return backend_base, synth_api_key


def _format_result(result: GEPACommandResult) -> str:
    """Format command result for display."""
    response = result.response or {}
    status = response.get("status", "unknown")
    lines = [f"Status: {status}"]

    if result.should_stop:
        lines.append(f"Should stop: {result.should_stop}")
    if result.termination_reason:
        lines.append(f"Termination reason: {result.termination_reason}")
    if result.checkpoint_requested:
        lines.append(f"Checkpoint requested: {result.checkpoint_requested}")

    # Add any additional response fields
    for key, value in response.items():
        if key != "status" and not key.startswith("_"):
            if isinstance(value, dict):
                lines.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                lines.append(f"{key}: {value}")

    return "\n".join(lines)


@click.group("job")
def job_group() -> None:
    """Control running GEPA jobs."""
    pass


@job_group.command("pause")
@click.argument("job_id")
@click.option(
    "--checkpoint/--no-checkpoint",
    default=True,
    help="Save checkpoint before pausing (default: yes)",
)
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def pause_command(
    job_id: str,
    checkpoint: bool,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Pause a running GEPA job.

    JOB_ID is the job identifier to pause.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Pausing job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Checkpoint: {checkpoint}")

    try:
        result = pause_job(job_id, backend_base, api_key, checkpoint=checkpoint)
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to pause job: {e}") from e


@job_group.command("cancel")
@click.argument("job_id")
@click.option(
    "--checkpoint/--no-checkpoint",
    default=False,
    help="Save checkpoint before canceling (default: no)",
)
@click.option("--reason", default=None, help="Cancellation reason")
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def cancel_command(
    job_id: str,
    checkpoint: bool,
    reason: str | None,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Cancel a running GEPA job.

    JOB_ID is the job identifier to cancel.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Canceling job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Checkpoint: {checkpoint}")
    if reason:
        click.echo(f"Reason: {reason}")

    try:
        result = cancel_job(
            job_id, backend_base, api_key, checkpoint=checkpoint, reason=reason
        )
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to cancel job: {e}") from e


@job_group.command("message")
@click.argument("job_id")
@click.argument("message")
@click.option(
    "--clear/--no-clear",
    default=False,
    help="Clear previous messages before adding this one",
)
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def message_command(
    job_id: str,
    message: str,
    clear: bool,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Send guidance to the prompt proposer.

    JOB_ID is the job identifier.
    MESSAGE is the guidance text to send (will be appended to proposer prompts).

    Example:
        synth-ai job message abc123 "Focus on concise instructions"
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Sending message to proposer for job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Message: {message}")
    click.echo(f"Clear previous: {clear}")

    try:
        result = message_proposer(
            job_id, message, backend_base, api_key, clear_previous=clear
        )
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to send message: {e}") from e


@job_group.command("extend-budget")
@click.argument("job_id")
@click.argument("additional_rollouts", type=int)
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def extend_budget_command(
    job_id: str,
    additional_rollouts: int,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Extend the rollout budget for a running GEPA job.

    JOB_ID is the job identifier.
    ADDITIONAL_ROLLOUTS is the number of rollouts to add.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Extending budget for job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Additional rollouts: {additional_rollouts}")

    try:
        result = extend_budget(job_id, additional_rollouts, backend_base, api_key)
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to extend budget: {e}") from e


@job_group.command("extend-generations")
@click.argument("job_id")
@click.argument("additional_generations", type=int)
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def extend_generations_command(
    job_id: str,
    additional_generations: int,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Extend the number of generations for a running GEPA job.

    JOB_ID is the job identifier.
    ADDITIONAL_GENERATIONS is the number of generations to add.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Extending generations for job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Additional generations: {additional_generations}")

    try:
        result = extend_generations(
            job_id, additional_generations, backend_base, api_key
        )
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to extend generations: {e}") from e


@job_group.command("checkpoint")
@click.argument("job_id")
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def checkpoint_command(
    job_id: str,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Force an immediate checkpoint for a running GEPA job.

    JOB_ID is the job identifier.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Requesting checkpoint for job {job_id}...")
    click.echo(f"Backend: {backend_base}")

    try:
        result = checkpoint_now(job_id, backend_base, api_key)
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to request checkpoint: {e}") from e


@job_group.command("ping")
@click.argument("job_id")
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def ping_command(
    job_id: str,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Ping a running GEPA job to check its status.

    JOB_ID is the job identifier.
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    click.echo(f"Pinging job {job_id}...")
    click.echo(f"Backend: {backend_base}")

    try:
        result = ping_job(job_id, backend_base, api_key)
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to ping job: {e}") from e


@job_group.command("config")
@click.argument("job_id")
@click.option(
    "--max-concurrent-rollouts",
    type=int,
    default=None,
    help="Update max_concurrent_rollouts",
)
@click.option(
    "--children-per-generation",
    type=int,
    default=None,
    help="Update children_per_generation",
)
@click.option(
    "--minibatch-size",
    type=int,
    default=None,
    help="Update minibatch_size",
)
@click.option("--backend", "backend_override", default=None, help="Backend base URL")
@click.option(
    "--env",
    "env_file",
    default=None,
    help=".env file to load",
)
def config_command(
    job_id: str,
    max_concurrent_rollouts: int | None,
    children_per_generation: int | None,
    minibatch_size: int | None,
    backend_override: str | None,
    env_file: str | None,
) -> None:
    """Update runtime configuration for a running GEPA job.

    JOB_ID is the job identifier.

    Example:
        synth-ai job config abc123 --max-concurrent-rollouts 10
    """
    backend_base, api_key = _get_backend_and_key(backend_override, env_file)

    # Build updates dict from provided options
    updates: dict[str, Any] = {}
    if max_concurrent_rollouts is not None:
        updates["max_concurrent_rollouts"] = max_concurrent_rollouts
    if children_per_generation is not None:
        updates["children_per_generation"] = children_per_generation
    if minibatch_size is not None:
        updates["minibatch_size"] = minibatch_size

    if not updates:
        raise click.ClickException(
            "No config updates provided. Use --help to see available options."
        )

    click.echo(f"Updating config for job {job_id}...")
    click.echo(f"Backend: {backend_base}")
    click.echo(f"Updates: {json.dumps(updates)}")

    try:
        result = update_config(job_id, updates, backend_base, api_key)
        click.echo("\nResult:")
        click.echo(_format_result(result))
    except Exception as e:
        raise click.ClickException(f"Failed to update config: {e}") from e


def register(cli: click.Group) -> None:
    """Register job control commands with the CLI."""
    cli.add_command(job_group)
