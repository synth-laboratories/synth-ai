"""Status and listing commands for the Synth CLI."""

from __future__ import annotations

import click

from .config import resolve_backend_config
from .subcommands.files import files_group
from .subcommands.jobs import jobs_group
from .subcommands.models import models_group
from .subcommands.runs import runs_group
from .subcommands.session import session_status_cmd
from .subcommands.summary import summary_command


def _attach_group(cli: click.Group, group: click.Group, name: str) -> None:
    """Attach the provided Click group to the CLI if not already present."""
    if name in cli.commands:
        return
    cli.add_command(group, name=name)


def register(cli: click.Group) -> None:
    """Register all status command groups on the provided CLI root."""

    @click.group(help="Inspect training jobs, models, files, and job runs.")
    @click.option(
        "--base-url",
        envvar="SYNTH_STATUS_BASE_URL",
        default=None,
        help="Synth backend base URL (defaults to environment configuration).",
    )
    @click.option(
        "--api-key",
        envvar="SYNTH_STATUS_API_KEY",
        default=None,
        help="API key for authenticated requests (falls back to Synth defaults).",
    )
    @click.option(
        "--timeout",
        default=30.0,
        show_default=True,
        type=float,
        help="HTTP request timeout in seconds.",
    )
    @click.pass_context
    def status(ctx: click.Context, base_url: str | None, api_key: str | None, timeout: float) -> None:
        """Populate shared backend configuration for subcommands."""
        cfg = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
        ctx.ensure_object(dict)
        ctx.obj["status_backend_config"] = cfg

    status.add_command(jobs_group, name="jobs")
    status.add_command(models_group, name="models")
    status.add_command(files_group, name="files")
    status.add_command(runs_group, name="runs")
    status.add_command(session_status_cmd, name="session")
    status.add_command(summary_command, name="summary")

    cli.add_command(status, name="status")
    _attach_group(cli, jobs_group, "jobs")
    _attach_group(cli, models_group, "models")
    _attach_group(cli, files_group, "files")
    _attach_group(cli, runs_group, "runs")
    if "status-summary" not in cli.commands:
        cli.add_command(summary_command, name="status-summary")
