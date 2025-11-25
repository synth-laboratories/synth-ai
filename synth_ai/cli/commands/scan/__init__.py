"""Scan command for discovering active task apps."""

from __future__ import annotations

import click

from synth_ai.cli.commands.scan.core import scan_command


def register(cli: click.Group) -> None:
    """Register the scan command with the CLI."""
    cli.add_command(scan_command)







