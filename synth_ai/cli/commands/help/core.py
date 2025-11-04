"""Help command implementation."""

import click
from click.exceptions import Exit

from . import COMMAND_HELP, get_command_help


@click.command("help")
@click.argument("command_name", type=str, required=False)
def help_command(command_name: str | None) -> None:
    """Display detailed help for Synth AI commands.
    
    USAGE
    -----
      uvx synth-ai help [COMMAND]
    
    EXAMPLES
    --------
      # List available help topics
      uvx synth-ai help
      
      # Get detailed help for deploy
      uvx synth-ai help deploy
      
      # Get detailed help for setup
      uvx synth-ai help setup
    """
    if not command_name:
        # Show list of available help topics
        click.echo("Synth AI - Detailed Help")
        click.echo("=" * 50)
        click.echo("\nAvailable help topics:")
        click.echo("")
        
        for cmd in sorted(COMMAND_HELP.keys()):
            click.echo(f"  • {cmd}")
        
        click.echo("\nUsage:")
        click.echo("  uvx synth-ai help [COMMAND]")
        click.echo("")
        click.echo("Examples:")
        click.echo("  uvx synth-ai help deploy")
        click.echo("  uvx synth-ai help setup")
        click.echo("")
        click.echo("You can also use standard --help flags:")
        click.echo("  uvx synth-ai deploy --help")
        click.echo("  uvx synth-ai setup --help")
        return
    
    # Show detailed help for specific command
    help_text = get_command_help(command_name)
    if not help_text:
        click.echo(f"No detailed help available for '{command_name}'", err=True)
        click.echo(f"\nTry: uvx synth-ai {command_name} --help", err=True)
        click.echo("Or: uvx synth-ai help (to see available topics)", err=True)
        raise Exit(1)
    
    click.echo(help_text)


def get_command() -> click.Command:
    """Get the help command for registration."""
    return help_command


def register(group: click.Group) -> None:
    """Register the help command with a Click group."""
    group.add_command(help_command)


__all__ = ["help_command", "get_command", "register"]
