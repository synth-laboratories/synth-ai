"""Task app CLI commands (serve, deploy, list, validate).

Commands for managing Synth task apps - local serving, Modal deployment,
validation, and discovery.
"""

from __future__ import annotations

import importlib

from synth_ai.cli.task_apps.commands import (
    AppChoice,
    TaskAppEntryType,
    _find_modal_executable,
    _is_modal_shim,
    _markov_message_from_dict,
    _modal_command_prefix,
    register,
    serve_command,
    task_app_group,
)
from synth_ai.cli.task_apps.main import task_app_cmd

# Re-export for backward compatibility
__all__ = [
    "AppChoice",
    "TaskAppEntryType",
    "task_app_cmd",
    "task_app_group",
    "serve_command",
    "register",
    "_find_modal_executable",
    "_is_modal_shim",
    "_modal_command_prefix",
    "_markov_message_from_dict",
    "importlib",
]
