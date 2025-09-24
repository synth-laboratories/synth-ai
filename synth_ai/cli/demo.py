#!/usr/bin/env python3
"""
CLI: interactive launcher for example demos and forwarders for new RL demo.

- `synth-ai demo` (no subcommand) -> legacy examples/ runner (run_demo.sh picker)
- `synth-ai demo deploy|configure|run` -> forwards to synth_ai.demos.core.cli
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click


def _find_demo_scripts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("run_demo.sh") if p.is_file()])


def _forward_to_new(args: list[str]) -> None:
    import sys
    try:
        from synth_ai.demos.core import cli as demo_cli  # type: ignore
    except Exception as e:  # pragma: no cover
        click.echo(f"Failed to import demo CLI: {e}")
        sys.exit(1)
    rc = int(demo_cli.main(args) or 0)
    if rc != 0:
        sys.exit(rc)


def register(cli):
    @cli.group("demo", invoke_without_command=True)
    @click.option("--list", "list_only", is_flag=True, help="List available legacy demos and exit")
    @click.option("-f", "filter_term", default="", help="Filter legacy demos by substring")
    @click.pass_context
    def demo(ctx: click.Context, list_only: bool, filter_term: str):
        """Demo helpers.

        - Legacy mode (no subcommand): find and run examples/*/run_demo.sh
        - New RL demo subcommands: deploy, configure, run
        """
        if ctx.invoked_subcommand is not None:
            return
        # Legacy behavior: interactive examples runner
        repo_root = Path(os.getcwd())
        examples_dir = repo_root / "examples"
        demos = _find_demo_scripts(examples_dir)
        if filter_term:
            demos = [p for p in demos if filter_term.lower() in str(p).lower()]

        if not demos:
            click.echo("No run_demo.sh scripts found under examples/.")
            return

        if list_only:
            click.echo("Available demos:")
            for p in demos:
                click.echo(f" - {p.relative_to(repo_root)}")
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

    # (prepare command removed; configure now prepares baseline TOML)

    @demo.command("deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy")
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option("--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)")
    def demo_deploy(local: bool, app: str | None, name: str, script: str | None):
        args: list[str] = ["rl_demo.deploy"]
        if local:
            args.append("--local")
        if app:
            args.extend(["--app", app])
        if name:
            args.extend(["--name", name])
        if script:
            args.extend(["--script", script])
        _forward_to_new(args)

    @demo.command("configure")
    def demo_configure():
        _forward_to_new(["rl_demo.configure"]) 

    @demo.command("run")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    def demo_run(batch_size: int | None, group_size: int | None, model: str | None, timeout: int):
        args = ["rl_demo.run"]
        if batch_size is not None:
            args.extend(["--batch-size", str(batch_size)])
        if group_size is not None:
            args.extend(["--group-size", str(group_size)])
        if model:
            args.extend(["--model", model])
        if timeout:
            args.extend(["--timeout", str(timeout)])
        _forward_to_new(args)
