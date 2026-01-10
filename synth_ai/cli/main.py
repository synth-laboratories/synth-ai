"""Root CLI group with lazy-loaded subcommands."""

from pathlib import Path

import click

from synth_ai.cli.lib.lazy import LazyGroup


def _discover_commands() -> dict[str, str]:
    """Auto-discover CLI command modules in synth_ai.cli."""
    cli_dir = Path(__file__).resolve().parent
    commands: dict[str, str] = {}
    for path in sorted(cli_dir.glob("*.py")):
        name = path.stem
        if name.startswith("_") or name in {"__init__", "__main__", "main"}:
            continue
        command_name = name.replace("_", "-")
        commands[command_name] = f"synth_ai.cli.{name}:{name}"
    return commands


def _get_version():
    try:
        from synth_ai import __version__

        return __version__
    except Exception:
        return "unknown"


@click.group(cls=LazyGroup, lazy_subcommands=_discover_commands(), invoke_without_command=True)
@click.version_option(version=_get_version(), prog_name="synth-ai")
@click.pass_context
def cli(ctx):
    """Synth AI CLI."""
    if ctx.invoked_subcommand is None:
        from synth_ai.tui import run_prompt_learning_tui

        run_prompt_learning_tui()
