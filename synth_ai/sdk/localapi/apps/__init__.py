"""LocalAPI app registry re-exports.

Prefer this module over synth_ai.sdk.task.apps.* moving forward.
"""

from __future__ import annotations

from synth_ai.sdk.task.apps import (
    LocalAPIEntry,
    ModalDeploymentConfig,
    TaskAppEntry,
    TaskAppRegistry,
    discover_task_apps_from_cwd,
    register_local_api,
    register_task_app,
    registry,
)

__all__ = [
    "LocalAPIEntry",
    "ModalDeploymentConfig",
    "TaskAppEntry",
    "TaskAppRegistry",
    "discover_task_apps_from_cwd",
    "register_local_api",
    "register_task_app",
    "registry",
]
