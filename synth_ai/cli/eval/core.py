from __future__ import annotations

import click
from synth_ai.cli.task_apps import eval_command as _eval_command

__all__ = ["command", "get_command"]

command = _eval_command


def get_command() -> click.Command:
    """Return the Click command implementing task-app evaluation."""
    return command
