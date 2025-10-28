"""Compatibility wrapper for task-app modal-serve command."""

from __future__ import annotations

from synth_ai.cli.task_apps import task_app_group

modal_serve_command = task_app_group.commands.get("modal-serve")
if modal_serve_command is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Modal-serve command is not registered on task_app_group")

__all__ = ["modal_serve_command"]
