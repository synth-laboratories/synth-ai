from __future__ import annotations

from collections.abc import Sequence

import click


def print_next_step_message(message: str, lines: Sequence[str]) -> None:
    """Display a consistently formatted 'next step' banner."""

    click.echo(f"\n➡️  Next, {message}:")
    for line in lines:
        click.echo(f"   {line}")
    click.echo('')
