"""LocalAPI server config re-exports.

Prefer this module over synth_ai.sdk.task.server.* moving forward.
"""

from __future__ import annotations

from synth_ai.sdk.task.server import (
    LocalAPIConfig,
    ProxyConfig,
    RubricBundle,
    TaskAppConfig,
    create_task_app,
    run_task_app,
)

create_local_api = create_task_app
run_local_api = run_task_app

__all__ = [
    "LocalAPIConfig",
    "TaskAppConfig",
    "ProxyConfig",
    "RubricBundle",
    "create_task_app",
    "create_local_api",
    "run_task_app",
    "run_local_api",
]
