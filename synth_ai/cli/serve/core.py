from __future__ import annotations

import click
from synth_ai.cli.task_apps import task_app_group

__all__ = ["command", "get_command"]

command = task_app_group.commands.get("serve")


def get_command() -> click.Command:
    if command is None:
        raise RuntimeError("Serve command is not registered on task_app_group")
    return command
