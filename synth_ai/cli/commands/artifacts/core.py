"""Core artifacts command group."""

from __future__ import annotations

import click

from .download import download_command
from .export import export_command
from .list import list_command
from .show import show_command


@click.group("artifacts", help="Manage artifacts (models and optimized prompts).")
def artifacts_group() -> None:
    """Artifacts command group for managing models and prompts."""
    pass


# Register subcommands
artifacts_group.add_command(list_command, name="list")
artifacts_group.add_command(export_command, name="export")
artifacts_group.add_command(download_command, name="download")
artifacts_group.add_command(show_command, name="show")

