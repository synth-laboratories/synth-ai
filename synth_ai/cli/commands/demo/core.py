"""Demo command - deprecated.

The demo apps have been removed. This command is no longer functional.
"""

import click

__all__ = ["command", "register"]


@click.group(
    "demo",
    invoke_without_command=True,
    help="[DEPRECATED] Demo helpers have been removed.",
)
@click.pass_context
def command(ctx: click.Context) -> None:
    """Demo command - deprecated."""
    click.echo("⚠️  Demo apps have been removed from synth-ai.")
    click.echo("")
    click.echo("For examples of task apps, see:")
    click.echo("  - https://docs.usesynth.ai/guides/local-api")
    click.echo("  - The examples/ directory in this repository")


@command.command("deploy")
def demo_deploy() -> None:
    """Deploy demo - deprecated."""
    click.echo("⚠️  Demo apps have been removed. This command is no longer available.")


@command.command("configure")
def demo_configure() -> None:
    """Configure demo - deprecated."""
    click.echo("⚠️  Demo apps have been removed. This command is no longer available.")


@command.command("setup")
def demo_setup() -> None:
    """Setup demo - deprecated."""
    click.echo("⚠️  Demo apps have been removed. This command is no longer available.")


@command.command("run")
def demo_run() -> None:
    """Run demo - deprecated."""
    click.echo("⚠️  Demo apps have been removed. This command is no longer available.")


def register(cli: click.Group) -> None:
    """Attach the demo command group to the CLI."""
    cli.add_command(command)
