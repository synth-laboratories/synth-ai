"""Compatibility wrapper for task-app deploy command."""

from __future__ import annotations

import click

from .task_apps import task_app_group

_deploy = task_app_group.commands.get("deploy")

if _deploy is None:
    raise RuntimeError("task_app_group does not define a 'deploy' command")

deploy_command: click.Command = _deploy

__all__ = ["deploy_command"]
