#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import click

from synth_ai.cli.demo.configure import run_configure
from synth_ai.cli.demo.deploy import run_deploy
from synth_ai.cli.demo.init import run_init
from synth_ai.cli.demo.run import run_job
from synth_ai.cli.setup import setup

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __pkg_version__ = _pkg_version("synth-ai")
    except PackageNotFoundError:
        try:
            from synth_ai import __version__ as __pkg_version__  # type: ignore
        except Exception:
            __pkg_version__ = "unknown"
except Exception:
    try:
        from synth_ai import __version__ as __pkg_version__  # type: ignore
    except Exception:
        __pkg_version__ = "unknown"

@click.group(
    help=f"Synth AI v{__pkg_version__} - Software for aiding the best and multiplying the will."
)
@click.version_option(version=__pkg_version__, prog_name="synth-ai")
def cli():
    """Top-level command group for Synth AI."""


@cli.command(name="setup")
def setup_command():
    """Perform SDK handshake and write keys to .env."""
    code = setup()
    if code:
        raise click.exceptions.Exit(code)


@cli.command(name="init")
@click.option("--template", type=str, default=None, help="Template id to instantiate")
@click.option("--dest", type=str, default=None, help="Destination directory for files")
@click.option("--force", is_flag=True, help="Overwrite existing files in destination")
def init_command(template: str | None, dest: str | None, force: bool):
    """Materialise a demo task app template into the current directory."""
    code = run_init(template=template, dest=dest, force=force)
    if code:
        raise click.exceptions.Exit(code)


@cli.command(name="deploy")
@click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
@click.option(
    "--app",
    type=click.Path(),
    default=None,
    help="Path to Modal app.py for uv run modal deploy",
)
@click.option("--name", type=str, default=None, help="Modal app name")
@click.option(
    "--script",
    type=click.Path(),
    default=None,
    help="Path to deploy_task_app.sh (optional legacy)",
)
def deploy_command(local: bool, app: str | None, name: str | None, script: str | None):
    """Deploy the currently configured demo task app."""
    code = run_deploy(local=local, app=app, name=name or "synth-math-demo", script=script)
    if code:
        raise click.exceptions.Exit(code)


@cli.command(name="run")
@click.option("--config", type=str, default=None, help="Path to TOML config (skip prompt)")
@click.option("--batch-size", type=int, default=None)
@click.option("--group-size", type=int, default=None)
@click.option("--model", type=str, default=None)
@click.option("--timeout", type=int, default=600)
@click.option("--dry-run", is_flag=True, help="Print request body and exit")
def run_command(
    config: str | None,
    batch_size: int | None,
    group_size: int | None,
    model: str | None,
    timeout: int,
    dry_run: bool,
):
    """Submit the configured RL job to the backend."""
    code = run_job(
        config=config,
        batch_size=batch_size,
        group_size=group_size,
        model=model,
        timeout=timeout,
        dry_run=dry_run,
    )
    if code:
        raise click.exceptions.Exit(code)


@cli.command(name="configure")
def configure_command():
    """Preflight demo environment and write configuration files."""
    code = run_configure()
    if code:
        raise click.exceptions.Exit(code)
