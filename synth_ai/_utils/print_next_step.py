from __future__ import annotations

from collections.abc import Sequence

import click

__all__ = ["print_next_step"]


def print_next_step(message: str, lines: Sequence[str]) -> None:
    click.echo(f"\n➡️  Next, {message}:")
    for line in lines:
        click.echo(f"   {line}")
    click.echo("")
