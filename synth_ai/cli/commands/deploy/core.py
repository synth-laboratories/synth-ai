from __future__ import annotations

import click
from synth_ai.cli.task_apps import task_app_group

__all__ = ["command", "get_command"]

command = task_app_group.commands.get("deploy")


def get_command() -> click.Command:
    if command is None:
        raise RuntimeError("Deploy command is not registered on task_app_group")
    return command
