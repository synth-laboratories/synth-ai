"""Compatibility wrapper for task-app modal-serve command."""



import click

from .task_apps import task_app_group

_modal_serve = task_app_group.commands.get("modal-serve")

if _modal_serve is None:
    raise RuntimeError("task_app_group does not define a 'modal-serve' command")

modal_serve_command: click.Command = _modal_serve

__all__ = ["modal_serve_command"]
