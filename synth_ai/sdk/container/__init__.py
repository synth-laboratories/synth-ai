"""Container SDK surface.

Prefer this module over synth_ai.sdk.container._impl.* moving forward. The task namespace
remains for backward compatibility while the naming transition completes.
"""

from typing import TYPE_CHECKING, Any

from .auth import ensure_container_auth

# Defer template imports to avoid circular dependency
# template.py imports from sdk.task, which may transitively import container

if TYPE_CHECKING:
    from synth_ai.sdk.container._impl import (
        ContainerClient,
        ContainerConfig,
        ContainerEndpoints,
        InProcessContainer,
        TaskInfo,
        create_container,
        run_container,
    )
    from synth_ai.sdk.optimization.internal.container_api import (
        ContainerHealth,
        check_container_health,
    )

    from .deploy import ContainerDeployResult, deploy_container
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
    create_container = create_container
    run_container = run_container

# Lazy imports for _impl symbols to avoid circular dependency
_IMPL_IMPORTS = {
    "InProcessContainer",
    "ContainerClient",
    "ContainerConfig",
    "ContainerEndpoints",
    "RubricBundle",
    "TaskInfo",
    "create_container",
    "run_container",
}
_TRAIN_IMPORTS = {
    "ContainerHealth",
    "check_container_health",
}
_LOCAL_IMPORTS = {
    "RolloutResponseBuilder",
}
_DEPLOY_IMPORTS = {
    "ContainerDeployResult",
    "deploy_container",
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
        from synth_ai.sdk.container import _impl

        return getattr(_impl, name)
    if name in _TRAIN_IMPORTS:
        from synth_ai.sdk.optimization.internal import container_api as _container_api

        return getattr(_container_api, name)
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
    if name == "create_container":
        from synth_ai.sdk.container._impl import create_container

        return create_container
    if name == "run_container":
        from synth_ai.sdk.container._impl import run_container

        return run_container
    if name in ("build_template_config", "create_template_app"):
        # Lazy import template functions to avoid circular dependency
        from .template import build_template_config, create_template_app

        if name == "build_template_config":
            return build_template_config
        if name == "create_template_app":
            return create_template_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core Container
    "ContainerClient",
    "ContainerConfig",
    "ContainerEndpoints",
    "ContainerHealth",
    "check_container_health",
    "InProcessContainer",
    "TaskInfo",
    "RubricBundle",
    "create_container",
    "create_container",
    "run_container",
    "run_container",
    "RolloutResponseBuilder",
    "ContainerDeployResult",
    "deploy_container",
    "ensure_container_auth",
    "build_template_config",
    "create_template_app",
    # Harbor integration
    "HarborExecutionBackend",
    "HarborExecutionError",
    "create_harbor_rollout_executor",
    "HarborInstanceProvider",
    "create_harbor_instance_provider",
]
