"""Root CLI group for Research and remaining local helpers."""

import click

from synth_ai.cli.dev_envs import dev_envs
from synth_ai.cli.research import research


def _get_version():
    try:
        from synth_ai import __version__

        return __version__
    except Exception:
        return "unknown"


@click.group(invoke_without_command=True)
@click.version_option(version=_get_version(), prog_name="synth-ai")
@click.pass_context
def cli(ctx):
    """Synth AI CLI."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(dev_envs)
cli.add_command(research)
