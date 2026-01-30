"""LocalAPI SDK surface.

Prefer this module over synth_ai.sdk.localapi._impl.* moving forward. The task namespace
remains for backward compatibility while the naming transition completes.
"""

from typing import TYPE_CHECKING, Any

from .auth import ensure_localapi_auth

# Defer template imports to avoid circular dependency
# template.py imports from sdk.task, which may transitively import localapi

if TYPE_CHECKING:
    from synth_ai.sdk.localapi._impl import (
        InProcessTaskApp,
        LocalAPIClient,
        LocalAPIConfig,
        LocalAPIEndpoints,
        TaskInfo,
        create_task_app,
        run_task_app,
    )
    from synth_ai.sdk.optimization.internal.local_api import LocalAPIHealth, check_local_api_health

    from .deploy import LocalAPIDeployResult, deploy_localapi
    from .harbor_adapter import (
        HarborExecutionBackend,
        HarborExecutionError,
        create_harbor_rollout_executor,
    )
    from .harbor_provider import (
        HarborInstanceProvider,
        create_harbor_instance_provider,
    )
    from .template import build_template_config, create_template_app

    # Type aliases for Pyright
    create_local_api = create_task_app
    run_local_api = run_task_app

# Lazy imports for _impl symbols to avoid circular dependency
_IMPL_IMPORTS = {
    "InProcessTaskApp",
    "LocalAPIClient",
    "LocalAPIConfig",
    "LocalAPIEndpoints",
    "RubricBundle",
    "TaskInfo",
    "create_task_app",
    "run_task_app",
}
_TRAIN_IMPORTS = {
    "LocalAPIHealth",
    "check_local_api_health",
}
_LOCAL_IMPORTS = {
    "RolloutResponseBuilder",
}
_DEPLOY_IMPORTS = {
    "LocalAPIDeployResult",
    "deploy_localapi",
}
_HARBOR_IMPORTS = {
    "HarborExecutionBackend",
    "HarborExecutionError",
    "create_harbor_rollout_executor",
    "HarborInstanceProvider",
    "create_harbor_instance_provider",
}


def __getattr__(name: str) -> Any:
    if name in _IMPL_IMPORTS:
        from synth_ai.sdk.localapi import _impl

        return getattr(_impl, name)
    if name in _TRAIN_IMPORTS:
        from synth_ai.sdk.optimization.internal import local_api as _local_api

        return getattr(_local_api, name)
    if name in _LOCAL_IMPORTS:
        from . import rollouts as _rollouts

        return getattr(_rollouts, name)
    if name in _DEPLOY_IMPORTS:
        from . import deploy as _deploy

        return getattr(_deploy, name)
    if name in _HARBOR_IMPORTS:
        # Harbor adapter and provider imports
        if name in (
            "HarborExecutionBackend",
            "HarborExecutionError",
            "create_harbor_rollout_executor",
        ):
            from . import harbor_adapter as _harbor_adapter

            return getattr(_harbor_adapter, name)
        if name in ("HarborInstanceProvider", "create_harbor_instance_provider"):
            from . import harbor_provider as _harbor_provider

            return getattr(_harbor_provider, name)
    if name == "create_local_api":
        from synth_ai.sdk.localapi._impl import create_task_app

        return create_task_app
    if name == "run_local_api":
        from synth_ai.sdk.localapi._impl import run_task_app

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
    # Core LocalAPI
    "LocalAPIClient",
    "LocalAPIConfig",
    "LocalAPIEndpoints",
    "LocalAPIHealth",
    "check_local_api_health",
    "InProcessTaskApp",
    "TaskInfo",
    "RubricBundle",
    "create_task_app",
    "create_local_api",
    "run_task_app",
    "run_local_api",
    "RolloutResponseBuilder",
    "LocalAPIDeployResult",
    "deploy_localapi",
    "ensure_localapi_auth",
    "build_template_config",
    "create_template_app",
    # Harbor integration
    "HarborExecutionBackend",
    "HarborExecutionError",
    "create_harbor_rollout_executor",
    "HarborInstanceProvider",
    "create_harbor_instance_provider",
]
