"""Task app serve command."""

from __future__ import annotations

import click

from .task_apps import _serve_cli, task_app_group


@click.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--reload/--no-reload", "reload_flag", default=False, help="Enable uvicorn auto-reload")
@click.option(
    "--force/--no-force",
    "force",
    default=False,
    help="Kill any process already bound to the selected port before starting",
)
@click.option(
    "--trace",
    "trace_dir",
    type=click.Path(),
    default=None,
    help="Enable tracing and write SFT JSONL files to this directory (default: traces/v3)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/v3/synth_ai.db)",
)
def serve_command(
    app_id: str | None,
    host: str,
    port: int | None,
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    """Top-level command to run a task app locally."""

    _serve_cli(app_id, host, port, reload_flag, force, trace_dir, trace_db)


@task_app_group.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--reload/--no-reload", "reload_flag", default=False, help="Enable uvicorn auto-reload")
@click.option(
    "--force/--no-force",
    "force",
    default=False,
    help="Kill any process already bound to the selected port before starting",
)
@click.option(
    "--trace",
    "trace_dir",
    type=click.Path(),
    default=None,
    help="Enable tracing and write SFT JSONL files to this directory (default: traces/v3)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/v3/synth_ai.db)",
)
def serve_task_group(
    app_id: str | None,
    host: str,
    port: int | None,
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    """Group subcommand to run a task app locally."""

    _serve_cli(app_id, host, port, reload_flag, force, trace_dir, trace_db)


__all__ = ["serve_command", "serve_task_group"]
