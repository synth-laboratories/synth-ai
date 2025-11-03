"""CLI command for baseline evaluation."""

from __future__ import annotations

from .core import command
from .list import list_command

__all__ = ["command"]

# Register list subcommand
command.add_command(list_command, name="list")

