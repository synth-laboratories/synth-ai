"""LocalAPI SDK surface.

Prefer this module over synth_ai.sdk.task.* moving forward. The task namespace
remains for backward compatibility while the naming transition completes.
"""

from __future__ import annotations

from synth_ai.sdk.api.train.local_api import LocalAPIHealth, check_local_api_health

_TASK_EXPORTS = {
    "InProcessTaskApp",
    "LocalAPIClient",
    "LocalAPIConfig",
    "LocalAPIEndpoints",
    "TaskInfo",
    "create_task_app",
    "run_task_app",
}


def _load_task_exports() -> dict[str, object]:
    try:
        from synth_ai import sdk as _sdk
        task_module = _sdk.task
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "synth_ai.sdk.localapi failed to import synth_ai.sdk.task. "
            "This module is required for LocalAPIConfig/create_local_api."
        ) from exc

    return {
        "InProcessTaskApp": task_module.InProcessTaskApp,
        "LocalAPIClient": task_module.LocalAPIClient,
        "LocalAPIConfig": task_module.LocalAPIConfig,
        "LocalAPIEndpoints": task_module.LocalAPIEndpoints,
        "TaskInfo": task_module.TaskInfo,
        "create_task_app": task_module.create_task_app,
        "run_task_app": task_module.run_task_app,
    }


def __getattr__(name: str):  # pragma: no cover - import-time behavior
    if name not in _TASK_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    exports = _load_task_exports()
    value = exports[name]
    globals()[name] = value
    return value


def create_local_api(*args, **kwargs):
    return __getattr__("create_task_app")(*args, **kwargs)


def run_local_api(*args, **kwargs):
    return __getattr__("run_task_app")(*args, **kwargs)
from .rollouts import RolloutResponseBuilder
from .auth import ensure_localapi_auth

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
    "ensure_localapi_auth",
    "build_template_config",
    "create_template_app",
]
