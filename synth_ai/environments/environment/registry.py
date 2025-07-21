from typing import Type, Dict, List

from synth_ai.environments.stateful.core import StatefulEnvironment

# Global registry for environment types
ENV_REGISTRY: Dict[str, Type[StatefulEnvironment]] = {}


def register_environment(name: str, cls: Type[StatefulEnvironment]) -> None:
    """Register an environment class under a unique name."""
    ENV_REGISTRY[name] = cls


def get_environment_cls(env_type: str) -> Type[StatefulEnvironment]:
    """Retrieve a registered environment class or raise an error."""
    try:
        return ENV_REGISTRY[env_type]
    except KeyError:
        raise ValueError(f"Unsupported environment type: {env_type}")


def list_supported_env_types() -> List[str]:
    """List all registered environment type names."""
    return list(ENV_REGISTRY.keys())
