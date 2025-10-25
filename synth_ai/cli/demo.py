#!/usr/bin/env python3
"""
CLI: interactive launcher for example demos and RL demo helpers.

- `synth-ai demo` (no subcommand) -> initialize RL demo files into ./synth_demo/
- `synth-ai demo deploy|configure|run` -> invoke RL demo helpers directly.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import click
from click.exceptions import Exit

demo_commands = cast(
    Any, importlib.import_module("synth_ai.demos.core.cli")
)


def _find_demo_scripts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("run_demo.sh") if p.is_file()])


def _run_demo_command(func, *args, **kwargs) -> None:
    """Invoke a demo command and exit via Click on non-zero status codes."""

    try:
        result = func(*args, **kwargs)
    except SystemExit as exc:  # pragma: no cover - defensive
        raise Exit(exc.code or 1) from exc

    if result is None:
        return

    try:
        code = int(result)
    except (TypeError, ValueError):
        return
    if code != 0:
        raise Exit(code)


def register(cli):
    @cli.group("demo", invoke_without_command=True)
    @click.option(
        "--force", is_flag=True, help="Overwrite existing files in CWD when initializing demo"
    )
    @click.option("--list", "list_only", is_flag=True, help="List available legacy demos and exit")
    @click.option("-f", "filter_term", default="", help="Filter legacy demos by substring")
    @click.pass_context
    def demo(ctx: click.Context, force: bool, list_only: bool, filter_term: str):
        """Demo helpers.

        - Default (no subcommand): initialize RL demo files into ./synth_demo/ (alias of rl_demo init)
        - Legacy mode: with --list, find and run examples/*/run_demo.sh
        - New RL demo subcommands: deploy, configure, run
        """
        if ctx.invoked_subcommand is not None:
            return

        # If explicitly asked to list legacy demos, show interactive picker
        if list_only:
            repo_root = Path(os.getcwd())
            examples_dir = repo_root / "examples"
            demos = _find_demo_scripts(examples_dir)
            if filter_term:
                demos = [p for p in demos if filter_term.lower() in str(p).lower()]

            if not demos:
                click.echo("No run_demo.sh scripts found under examples/.")
                return

            click.echo("Available demos:")
            for idx, p in enumerate(demos, start=1):
                click.echo(f" {idx}. {p.relative_to(repo_root)}")
            click.echo("")

            def _validate_choice(val: str) -> int:
                try:
                    i = int(val)
                except Exception as err:
                    raise click.BadParameter("Enter a number from the list") from err
                if i < 1 or i > len(demos):
                    raise click.BadParameter(f"Choose a number between 1 and {len(demos)}")
                return i

            choice = click.prompt("Select a demo to run", value_proc=_validate_choice)
            script = demos[choice - 1]

            click.echo("")
            click.echo(f"🚀 Running {script.relative_to(repo_root)}\n")

            try:
                subprocess.run(["bash", str(script)], check=True)
            except subprocess.CalledProcessError as e:
                click.echo(f"❌ Demo exited with non-zero status: {e.returncode}")
            except KeyboardInterrupt:
                click.echo("\n🛑 Demo interrupted by user")
            return

        # Default: initialize RL demo files via new command
        _run_demo_command(demo_commands.init, force=force)

    # (prepare command removed; configure now prepares baseline TOML)

    # Help pyright understand dynamic Click group attributes
    _dg = cast(Any, demo)

    @_dg.command("deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option(
        "--app",
        type=click.Path(),
        default=None,
        help="Path to Modal app.py for uv run modal deploy",
    )
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option(
        "--script",
        type=click.Path(),
        default=None,
        help="Path to deploy_task_app.sh (optional legacy)",
    )
    def demo_deploy(local: bool, app: str | None, name: str, script: str | None):
        _run_demo_command(
            demo_commands.deploy,
            local=local,
            app=app,
            name=name,
            script=script,
        )

    @_dg.command("configure")
    def demo_configure():
        _run_demo_command(demo_commands.run)

    @_dg.command("setup")
    def demo_setup():
        _run_demo_command(demo_commands.setup)

    @_dg.command("run")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    def demo_run(batch_size: int | None, group_size: int | None, model: str | None, timeout: int):
        _run_demo_command(
            demo_commands.run,
            batch_size=batch_size,
            group_size=group_size,
            model=model,
            timeout=timeout,
        )

    @cli.command("setup")
    def setup_alias():
        """Perform SDK handshake and write keys to .env."""
        _run_demo_command(demo_commands.setup)
