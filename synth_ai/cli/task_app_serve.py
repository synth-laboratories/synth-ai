"""Compatibility wrapper for task-app serve command."""

from __future__ import annotations

from synth_ai.cli.task_apps import serve_command, task_app_group

serve_task_group = task_app_group.commands.get("serve")
if serve_task_group is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Serve command is not registered on task_app_group")

__all__ = ["serve_command", "serve_task_group"]
