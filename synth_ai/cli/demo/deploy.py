from __future__ import annotations

import click

from .common import forward_to_core


def register(group):
    @group.command("deploy")
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
        args: list[str] = ["demo.deploy"]
        if local:
            args.append("--local")
        if app:
            args.extend(["--app", app])
        if name:
            args.extend(["--name", name])
        if script:
            args.extend(["--script", script])
        forward_to_core(args)
