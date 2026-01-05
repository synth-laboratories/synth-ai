"""LocalAPI SDK surface.

Prefer this module over synth_ai.sdk.task.* moving forward. The task namespace
remains for backward compatibility while the naming transition completes.
"""

from typing import TYPE_CHECKING

from synth_ai.sdk.api.train.local_api import LocalAPIHealth, check_local_api_health
from .rollouts import RolloutResponseBuilder
from .auth import ensure_localapi_auth
# Defer template imports to avoid circular dependency
# template.py imports from sdk.task, which may transitively import localapi

if TYPE_CHECKING:
    from synth_ai.sdk.task import (
        InProcessTaskApp,
        LocalAPIClient,
        LocalAPIConfig,
        LocalAPIEndpoints,
        TaskInfo,
        create_task_app,
        run_task_app,
    )
    from .template import build_template_config, create_template_app
    
    # Type aliases for Pyright
    create_local_api = create_task_app
    run_local_api = run_task_app

# Lazy imports for sdk.task symbols to avoid circular dependency
# The chain is: sdk.task -> in_process_runner -> prompt_learning -> localapi.auth
# When auth is imported, this __init__.py runs, so we must defer sdk.task imports
_TASK_IMPORTS = {
    "InProcessTaskApp",
    "LocalAPIClient",
    "LocalAPIConfig",
    "LocalAPIEndpoints",
    "TaskInfo",
    "create_task_app",
    "run_task_app",
}


def __getattr__(name: str):
    if name in _TASK_IMPORTS:
        from synth_ai.sdk import task

        return getattr(task, name)
    if name == "create_local_api":
        from synth_ai.sdk.task import create_task_app

        return create_task_app
    if name == "run_local_api":
        from synth_ai.sdk.task import run_task_app

        return run_task_app
    if name in ("build_template_config", "create_template_app"):
        # Lazy import template functions to avoid circular dependency
        from .template import build_template_config, create_template_app
        
        if name == "build_template_config":
            return build_template_config
        if name == "create_template_app":
            return create_template_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
