"""Task app serve command."""

from __future__ import annotations

from collections.abc import Sequence

import click

from .task_apps import _serve_cli, task_app_group


@click.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--env-file", multiple=True, type=click.Path(), help="Extra .env files to load")
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
    help="Enable tracing and write SFT JSONL files to this directory (default: traces)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/task_app_traces_<timestamp>.db)",
)
def serve_command(
    app_id: str | None,
    host: str,
    port: int | None,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    """Top-level command to run a task app locally."""

    _serve_cli(app_id, host, port, env_file, reload_flag, force, trace_dir, trace_db)


@task_app_group.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--env-file", multiple=True, type=click.Path(), help="Extra .env files to load")
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
    help="Enable tracing and write SFT JSONL files to this directory (default: traces)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/task_app_traces_<timestamp>.db)",
)
def serve_task_group(
    app_id: str | None,
    host: str,
    port: int | None,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    """Group subcommand to run a task app locally."""

    _serve_cli(app_id, host, port, env_file, reload_flag, force, trace_dir, trace_db)


__all__ = ["serve_command", "serve_task_group"]
