"""Task app CLI commands (serve, deploy, list, validate).

Commands for managing Synth task apps - local serving, Modal deployment,
validation, and discovery.
"""

from synth_ai.cli.task_apps.main import task_app_cmd
from synth_ai.cli.task_apps.commands import (
    task_app_group,
    serve_command,
    register,
)

# Re-export for backward compatibility
__all__ = [
    "task_app_cmd",
    "task_app_group",
    "serve_command",
    "register",
]

