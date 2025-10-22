#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import sys

import click

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


# === Demo command group (bridges to demos.core CLI) ===
@cli.group()
def demo():
    """Demo helpers (deploy, configure, run)."""


def _forward_to_demo(args: list[str]) -> None:
    # Lazy import to avoid loading demo deps unless needed
    try:
        from synth_ai.demos.core import cli as demo_cli  # type: ignore
    except Exception as e:  # pragma: no cover
        click.echo(f"Failed to import demo CLI: {e}")
        sys.exit(1)
    rc = int(demo_cli.main(args) or 0)  # type: ignore[attr-defined]
    if rc != 0:
        sys.exit(rc)


# (prepare command removed; handled by configure)


@demo.command()
@click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
@click.option(
    "--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy"
)
@click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
@click.option(
    "--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)"
)
def deploy(local: bool, app: str | None, name: str, script: str | None):
    """Deploy the Math Task App (Modal by default)."""
    args: list[str] = ["demo.deploy"]
    if local:
        args.append("--local")
    if app:
        args.extend(["--app", app])
    if name:
        args.extend(["--name", name])
    if script:
        args.extend(["--script", script])
    _forward_to_demo(args)


@demo.command()
def configure():
    """Print resolved environment and config path."""
    _forward_to_demo(["demo.configure"])


@demo.command()
def setup():
    """Perform SDK handshake and write keys to .env."""
    _forward_to_demo(["demo.setup"])


@demo.command()
@click.option("--template", type=str, default=None, help="Template id to instantiate")
@click.option("--dest", type=str, default=None, help="Destination directory for files")
@click.option("--force", is_flag=True, help="Overwrite existing files in destination")
def init(template: str | None, dest: str | None, force: bool):
    """Copy demo task app template into the current directory."""
    args: list[str] = ["demo.init"]
    if template:
        args.extend(["--template", template])
    if dest:
        args.extend(["--dest", dest])
    if force:
        args.append("--force")
    _forward_to_demo(args)


@demo.command()
@click.option("--batch-size", type=int, default=None)
@click.option("--group-size", type=int, default=None)
@click.option("--model", type=str, default=None)
@click.option("--timeout", type=int, default=600)
def run(batch_size: int | None, group_size: int | None, model: str | None, timeout: int):
    """Kick off a short RL job using the prepared TOML."""
    args = ["run"]
    if batch_size is not None:
        args.extend(["--batch-size", str(batch_size)])
    if group_size is not None:
        args.extend(["--group-size", str(group_size)])
    if model:
        args.extend(["--model", model])
    if timeout:
        args.extend(["--timeout", str(timeout)])
    _forward_to_demo(args)


@cli.command(name="setup")
def setup_command():
    """Perform SDK handshake and write keys to .env."""
    _forward_to_demo(["demo.setup"])
