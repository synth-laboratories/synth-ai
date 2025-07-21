# This file re-exports the actual registry functions from synth_ai.environments.environment.registry
# to be used by the service layer, maintaining a clean separation if needed.
from synth_ai.environments.environment.registry import (
    register_environment,
    get_environment_cls,
    list_supported_env_types,
)

__all__ = ["register_environment", "get_environment_cls", "list_supported_env_types"]
