import importlib
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import click
from click.exceptions import Exit

__all__ = ["command", "register"]

_demo_cli = cast(Any, importlib.import_module("synth_ai.cli.demo_apps.core.cli"))


def _find_demo_scripts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("run_demo.sh") if path.is_file())


def _run_demo_command(func: Any, *args: Any, **kwargs: Any) -> None:
    """Invoke a demo command and map non-zero exits to Click exits."""

    try:
        result = func(*args, **kwargs)
    except SystemExit as exc:  # pragma: no cover - defensive shim
        raise Exit(exc.code if isinstance(exc.code, int) else 1) from exc

    if result is None:
        return

    try:
        code = int(result)
    except (TypeError, ValueError):
        return
    if code != 0:
        raise Exit(code)


@click.group(
    "demo",
    invoke_without_command=True,
    help="Demo helpers for the math RL pipeline.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files in the current directory when initializing demo assets.",
)
@click.option("--list", "list_only", is_flag=True, help="List available legacy demos and exit.")
@click.option("-f", "filter_term", default="", help="Filter legacy demos by substring.")
@click.pass_context
def command(ctx: click.Context, force: bool, list_only: bool, filter_term: str) -> None:
    """Default command: initialize RL demo files into ./synth_demo/ (alias of `demo init`)."""
    if ctx.invoked_subcommand is not None:
        return

    if list_only:
        repo_root = Path(os.getcwd())
        examples_dir = repo_root / "examples"
        demos = _find_demo_scripts(examples_dir)
        if filter_term:
            term = filter_term.lower()
            demos = [path for path in demos if term in str(path).lower()]

        if not demos:
            click.echo("No run_demo.sh scripts found under examples/.")
            return

        click.echo("Available demos:")
        for idx, path in enumerate(demos, start=1):
            click.echo(f" {idx}. {path.relative_to(repo_root)}")
        click.echo("")

        def _validate_choice(val: str) -> int:
            try:
                selection = int(val)
            except Exception as err:  # pragma: no cover - Click handles prompt errors
                raise click.BadParameter("Enter a number from the list") from err
            if selection < 1 or selection > len(demos):
                raise click.BadParameter(f"Choose a number between 1 and {len(demos)}")
            return selection

        choice = click.prompt("Select a demo to run", value_proc=_validate_choice)
        script = demos[choice - 1]

        click.echo("")
        click.echo(f"🚀 Running {script.relative_to(repo_root)}\n")

        try:
            subprocess.run(["bash", str(script)], check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(f"❌ Demo exited with non-zero status: {exc.returncode}")
        except KeyboardInterrupt:
            click.echo("\n🛑 Demo interrupted by user")
        return

    _run_demo_command(_demo_cli.init, force=force)


@command.command("deploy")
@click.option("--local", is_flag=True, help="Run the local FastAPI app instead of deploying to Modal.")
@click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for `uv run modal deploy`.")
@click.option("--name", type=str, default="synth-math-demo", help="Modal app name.")
@click.option(
    "--script",
    type=click.Path(),
    default=None,
    help="Path to deploy_task_app.sh (optional legacy helper).",
)
def demo_deploy(local: bool, app: str | None, name: str, script: str | None) -> None:
    _run_demo_command(
        _demo_cli.deploy,
        local=local,
        app=app,
        name=name,
        script=script,
    )


@command.command("configure")
def demo_configure() -> None:
    _run_demo_command(_demo_cli.run)


@command.command("setup")
def demo_setup() -> None:
    _run_demo_command(_demo_cli.setup)


@command.command("run")
@click.option("--batch-size", type=int, default=None)
@click.option("--group-size", type=int, default=None)
@click.option("--model", type=str, default=None)
@click.option("--timeout", type=int, default=600)
def demo_run(
    batch_size: int | None,
    group_size: int | None,
    model: str | None,
    timeout: int,
) -> None:
    _run_demo_command(
        _demo_cli.run,
        batch_size=batch_size,
        group_size=group_size,
        model=model,
        timeout=timeout,
    )


def register(cli: click.Group) -> None:
    """Attach the demo command group and related aliases to the CLI."""
    cli.add_command(command)
