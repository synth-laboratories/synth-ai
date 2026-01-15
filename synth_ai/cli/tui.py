"""TUI command."""

import click


@click.command()
def tui() -> None:
    """Launch the TUI."""
    from synth_ai.tui import run_tui

    run_tui()
