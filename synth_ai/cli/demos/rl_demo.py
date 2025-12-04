#!/usr/bin/env python3
"""
New RL demo command group kept fully separate from legacy demo.

Usage examples:
  uvx synth-ai rl_demo check
  uvx synth-ai rl_demo deploy --app /path/to/math_task_app.py --name synth-math-demo
  uvx synth-ai rl_demo configure
  uvx synth-ai rl_demo run --batch-size 4 --group-size 16 --model Qwen/Qwen3-0.6B
  uvx synth-ai run --config demo_config.toml

For convenience, dotted aliases are also exposed:
  uvx synth-ai rl_demo.check
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import click
from click.exceptions import Exit

demo_commands = cast(Any, importlib.import_module("synth_ai.cli.demo_apps.core.cli"))


def _run_demo_command(func, *args, **kwargs) -> None:
    """Invoke a demo command and exit via Click on non-zero status codes."""

    try:
        result = func(*args, **kwargs)
    except SystemExit as exc:  # pragma: no cover - defensive
        raise Exit(exc.code if isinstance(exc.code, int) else 1) from exc

    if result is None:
        return
    try:
        code = int(result)
    except (TypeError, ValueError):
        return
    if code != 0:
        raise Exit(code)


def register(cli):
    @cli.group("rl_demo")
    def rl_demo():
        """RL Demo commands (separate from legacy demo)."""

    # Help pyright understand dynamic Click group attributes
    _rlg = cast(Any, rl_demo)

    @_rlg.command("setup")
    def rl_setup():
        _run_demo_command(demo_commands.setup)

    # (prepare command removed; consolidated into configure)

    @_rlg.command("deploy")
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
    def rl_deploy(local: bool, app: str | None, name: str, script: str | None):
        _run_demo_command(
            demo_commands.deploy,
            local=local,
            app=app,
            name=name,
            script=script,
        )

    @_rlg.command("configure")
    def rl_configure():
        _run_demo_command(demo_commands.run)

    @_rlg.command("init")
    @click.option("--template", type=str, default=None, help="Template id to instantiate")
    @click.option("--dest", type=click.Path(), default=None, help="Destination directory for files")
    @click.option("--force", is_flag=True, help="Overwrite existing files in destination")
    def rl_init(template: str | None, dest: str | None, force: bool):
        _run_demo_command(
            demo_commands.init,
            template=template,
            dest=dest,
            force=force,
        )

    @_rlg.command("run")
    @click.option(
        "--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)"
    )
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def rl_run(
        config: str | None,
        batch_size: int | None,
        group_size: int | None,
        model: str | None,
        timeout: int,
        dry_run: bool,
    ):
        _run_demo_command(
            demo_commands.run,
            config=config,
            batch_size=batch_size,
            group_size=group_size,
            model=model,
            timeout=timeout,
            dry_run=dry_run,
        )

    # Dotted aliases (top-level): legacy check → setup
    @cli.command("rl_demo.check")
    def rl_check_alias():
        _run_demo_command(demo_commands.setup)

    @cli.command("rl_demo.setup")
    def rl_setup_alias():
        _run_demo_command(demo_commands.setup)

    # (prepare alias removed)

    @cli.command("rl_demo.deploy")
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
    def rl_deploy_alias(local: bool, app: str | None, name: str, script: str | None):
        _run_demo_command(
            demo_commands.deploy,
            local=local,
            app=app,
            name=name,
            script=script,
        )

    @cli.command("rl_demo.configure")
    def rl_configure_alias():
        _run_demo_command(demo_commands.run)

    @cli.command("rl_demo.init")
    @click.option("--template", type=str, default=None, help="Template id to instantiate")
    @click.option("--dest", type=click.Path(), default=None, help="Destination directory for files")
    @click.option("--force", is_flag=True, help="Overwrite existing files in destination")
    def rl_init_alias(template: str | None, dest: str | None, force: bool):
        _run_demo_command(
            demo_commands.init,
            template=template,
            dest=dest,
            force=force,
        )

    @cli.command("rl_demo.run")
    @click.option(
        "--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)"
    )
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def rl_run_alias(
        config: str | None,
        batch_size: int | None,
        group_size: int | None,
        model: str | None,
        timeout: int,
        dry_run: bool,
    ):
        _run_demo_command(
            demo_commands.run,
            config=config,
            batch_size=batch_size,
            group_size=group_size,
            model=model,
            timeout=timeout,
            dry_run=dry_run,
        )

    # Top-level convenience alias: `synth-ai deploy`
    @cli.command("demo-deploy")
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
    def deploy_demo(local: bool, app: str | None, name: str, script: str | None):
        _run_demo_command(
            demo_commands.deploy,
            local=local,
            app=app,
            name=name,
            script=script,
        )

    @cli.command("run")
    @click.option(
        "--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)"
    )
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def run_top(
        config: str | None,
        batch_size: int | None,
        group_size: int | None,
        model: str | None,
        timeout: int,
        dry_run: bool,
    ):
        _run_demo_command(
            demo_commands.run,
            config=config,
            batch_size=batch_size,
            group_size=group_size,
            model=model,
            timeout=timeout,
            dry_run=dry_run,
        )
