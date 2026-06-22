"""Root CLI group with Python-only containers platform commands."""

import click

from synth_ai.cli.containers import containers
from synth_ai.cli.pools import pools
from synth_ai.cli.tunnels import tunnels


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
    """Synth AI Python-only containers platform."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(containers)
cli.add_command(pools)
cli.add_command(tunnels)
