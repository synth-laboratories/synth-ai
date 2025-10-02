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

import click


def _forward(args: list[str]) -> None:
    import sys
    try:
        from synth_ai.demos.core import cli as demo_cli  # type: ignore
    except Exception as e:  # pragma: no cover
        click.echo(f"Failed to import RL demo CLI: {e}")
        sys.exit(1)
    rc = int(demo_cli.main(args) or 0)
    if rc != 0:
        sys.exit(rc)


def register(cli):
    @cli.group("rl_demo")
    def rl_demo():
        """RL Demo commands (separate from legacy demo)."""

    # Help pyright understand dynamic Click group attributes
    from typing import Any, cast as _cast
    _rlg = _cast(Any, rl_demo)

    @_rlg.command("setup")
    def rl_setup():
        _forward(["rl_demo.setup"])  # primary setup command

    # (prepare command removed; consolidated into configure)

    @_rlg.command("deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy")
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option("--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)")
    def rl_deploy(local: bool, app: str | None, name: str, script: str | None):
        args: list[str] = ["rl_demo.deploy"]
        if local:
            args.append("--local")
        if app:
            args.extend(["--app", app])
        if name:
            args.extend(["--name", name])
        if script:
            args.extend(["--script", script])
        _forward(args)

    @_rlg.command("configure")
    def rl_configure():
        _forward(["rl_demo.configure"]) 

    @_rlg.command("init")
    @click.option("--force", is_flag=True, help="Overwrite existing files in CWD")
    def rl_init(force: bool):
        args = ["rl_demo.init"]
        if force:
            args.append("--force")
        _forward(args)

    @_rlg.command("run")
    @click.option("--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def rl_run(config: str | None, batch_size: int | None, group_size: int | None, model: str | None, timeout: int, dry_run: bool):
        args = ["rl_demo.run"]
        if config:
            args.extend(["--config", config])
        if batch_size is not None:
            args.extend(["--batch-size", str(batch_size)])
        if group_size is not None:
            args.extend(["--group-size", str(group_size)])
        if model:
            args.extend(["--model", model])
        if timeout is not None:
            args.extend(["--timeout", str(timeout)])
        if dry_run:
            args.append("--dry-run")
        _forward(args)

    # Dotted aliases (top-level): legacy check → setup
    @cli.command("rl_demo.check")
    def rl_check_alias():
        _forward(["rl_demo.setup"]) 

    @cli.command("rl_demo.setup")
    def rl_setup_alias():
        _forward(["rl_demo.setup"]) 

    # (prepare alias removed)

    @cli.command("rl_demo.deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy")
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option("--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)")
    def rl_deploy_alias(local: bool, app: str | None, name: str, script: str | None):
        args: list[str] = ["rl_demo.deploy"]
        if local:
            args.append("--local")
        if app:
            args.extend(["--app", app])
        if name:
            args.extend(["--name", name])
        if script:
            args.extend(["--script", script])
        _forward(args)

    @cli.command("rl_demo.configure")
    def rl_configure_alias():
        _forward(["rl_demo.configure"]) 

    @cli.command("rl_demo.init")
    @click.option("--force", is_flag=True, help="Overwrite existing files in CWD")
    def rl_init_alias(force: bool):
        args = ["rl_demo.init"]
        if force:
            args.append("--force")
        _forward(args)

    @cli.command("rl_demo.run")
    @click.option("--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def rl_run_alias(config: str | None, batch_size: int | None, group_size: int | None, model: str | None, timeout: int, dry_run: bool):
        args = ["rl_demo.run"]
        if config:
            args.extend(["--config", config])
        if batch_size is not None:
            args.extend(["--batch-size", str(batch_size)])
        if group_size is not None:
            args.extend(["--group-size", str(group_size)])
        if model:
            args.extend(["--model", model])
        if timeout is not None:
            args.extend(["--timeout", str(timeout)])
        if dry_run:
            args.append("--dry-run")
        _forward(args)

    # Top-level convenience alias: `synth-ai deploy`
    @cli.command("deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy")
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option("--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)")
    def deploy_top(local: bool, app: str | None, name: str, script: str | None):
        args: list[str] = ["rl_demo.deploy"]
        if local:
            args.append("--local")
        if app:
            args.extend(["--app", app])
        if name:
            args.extend(["--name", name])
        if script:
            args.extend(["--script", script])
        _forward(args)

    @cli.command("run")
    @click.option("--config", type=click.Path(), default=None, help="Path to TOML config (skip prompt)")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def run_top(config: str | None, batch_size: int | None, group_size: int | None, model: str | None, timeout: int, dry_run: bool):
        args = ["run"]
        if config:
            args.extend(["--config", config])
        if batch_size is not None:
            args.extend(["--batch-size", str(batch_size)])
        if group_size is not None:
            args.extend(["--group-size", str(group_size)])
        if model:
            args.extend(["--model", model])
        if timeout is not None:
            args.extend(["--timeout", str(timeout)])
        if dry_run:
            args.append("--dry-run")
        _forward(args)
