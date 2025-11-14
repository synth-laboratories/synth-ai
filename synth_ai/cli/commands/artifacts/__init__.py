"""Artifacts CLI commands for managing models and prompts."""

from __future__ import annotations

import click


def register(cli: click.Group) -> None:
    """Register artifacts commands with the main CLI."""
    from .core import artifacts_group
    
    cli.add_command(artifacts_group, name="artifacts")

