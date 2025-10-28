"""Compatibility wrapper for task-app serve command."""



import click

from .task_apps import serve_command as task_app_serve_command
from .task_apps import task_app_group

serve_command = task_app_serve_command

_group_serve = task_app_group.commands.get("serve")
if _group_serve is None:
    raise RuntimeError("task_app_group does not define a 'serve' command")

serve_task_group: click.Command = _group_serve

__all__ = ["serve_command", "serve_task_group"]
