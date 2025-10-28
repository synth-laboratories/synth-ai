from __future__ import annotations

import click
from synth_ai.cli.task_apps import filter_command as _filter_command

__all__ = ["command", "get_command"]

command = _filter_command


def get_command() -> click.Command:
    """Return the Click command implementing dataset filtering."""
    return command
