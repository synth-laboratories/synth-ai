"""LocalAPI SDK surface.

Prefer this module over synth_ai.sdk.task.* moving forward. The task namespace
remains for backward compatibility while the naming transition completes.
"""

from __future__ import annotations

from synth_ai.sdk.api.train.local_api import LocalAPIHealth, check_local_api_health
from synth_ai.sdk.task import (
    InProcessTaskApp,
    LocalAPIClient,
    LocalAPIConfig,
    LocalAPIEndpoints,
    TaskInfo,
    create_task_app,
    run_task_app,
)
from .rollouts import RolloutResponseBuilder

create_local_api = create_task_app
run_local_api = run_task_app
from .template import build_template_config, create_template_app

__all__ = [
    "LocalAPIClient",
    "LocalAPIConfig",
    "LocalAPIEndpoints",
    "LocalAPIHealth",
    "check_local_api_health",
    "InProcessTaskApp",
    "TaskInfo",
    "create_task_app",
    "create_local_api",
    "run_task_app",
    "run_local_api",
    "RolloutResponseBuilder",
    "build_template_config",
    "create_template_app",
]
