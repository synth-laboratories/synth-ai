"""Compatibility wrapper for task-app serve command."""

from __future__ import annotations

from synth_ai.cli.commands.serve import command as serve_task_group
from synth_ai.cli.task_apps import serve_command

__all__ = ["serve_command", "serve_task_group"]
