"""
Environment Registry Module

This module provides a global registry system for environment types in the Synth AI framework.
The registry allows environments to be registered by name and retrieved dynamically,
enabling flexible environment management and discovery.

The registry supports:
- Dynamic environment registration at runtime
- Type-safe environment retrieval
- Environment discovery and listing
- Centralized environment management

Example:
    >>> from synth_ai.environments.environment.registry import register_environment
    >>> from myproject.environments import MyCustomEnvironment

    >>> # Register a custom environment
    >>> register_environment("my_env", MyCustomEnvironment)

    >>> # List available environments
    >>> available_envs = list_supported_env_types()
    >>> print("Available environments:", available_envs)

    >>> # Get environment class for instantiation
    >>> env_cls = get_environment_cls("my_env")
    >>> env_instance = env_cls(task_config)
"""

from typing import Type, Dict, List
import logging
import importlib.metadata

from synth_ai.environments.stateful.core import StatefulEnvironment

logger = logging.getLogger(__name__)

# Global registry for environment types
ENV_REGISTRY: Dict[str, Type[StatefulEnvironment]] = {}


def register_environment(name: str, cls: Type[StatefulEnvironment]) -> None:
    """
    Register an environment class under a unique name.

    This function adds an environment class to the global registry, making it
    available for dynamic instantiation by name. This is particularly useful
    for building flexible systems where environment types are determined at
    runtime or configured through external settings.

    Args:
        name: Unique identifier for the environment. This name will be used
            to retrieve the environment class later. Names should be descriptive
            and follow a consistent naming convention (e.g., "CartPole", "Sokoban").
        cls: The environment class to register. Must be a subclass of
            StatefulEnvironment and implement all required abstract methods.

    Raises:
        TypeError: If cls is not a subclass of StatefulEnvironment
        ValueError: If name is empty or None

    Example:
        >>> class MyGameEnvironment(StatefulEnvironment):
        ...     # Implementation of abstract methods
        ...     pass

        >>> register_environment("my_game", MyGameEnvironment)
        >>>
        >>> # Now the environment can be retrieved by name
        >>> env_cls = get_environment_cls("my_game")
        >>> env = env_cls(task_config)

    Note:
        Environment names are case-sensitive. It's recommended to use
        consistent naming conventions (e.g., lowercase with underscores
        or CamelCase) across your application.
    """
    ENV_REGISTRY[name] = cls


def get_environment_cls(env_type: str) -> Type[StatefulEnvironment]:
    """
    Retrieve a registered environment class by name.

    This function looks up an environment class in the global registry
    and returns it for instantiation. This enables dynamic environment
    creation based on string identifiers, which is useful for:
    - Configuration-driven environment selection
    - API endpoints that accept environment type parameters
    - Plugin systems and modular architectures
    - Testing frameworks that need to test multiple environment types

    Args:
        env_type: The name of the environment type to retrieve. Must match
            a name that was previously registered using register_environment().

    Returns:
        Type[StatefulEnvironment]: The environment class that can be instantiated
            with appropriate task parameters.

    Raises:
        ValueError: If env_type is not found in the registry. The error message
            will include the invalid type name and suggest checking available types.

    Example:
        >>> # Retrieve and instantiate an environment
        >>> env_cls = get_environment_cls("CartPole")
        >>> environment = env_cls(task_instance)
        >>>
        >>> # Use in configuration-driven scenarios
        >>> config = {"environment_type": "Sokoban", "difficulty": "easy"}
        >>> env_cls = get_environment_cls(config["environment_type"])
        >>> env = env_cls(create_task(config))

    See Also:
        list_supported_env_types(): Get all available environment type names
        register_environment(): Register new environment types
    """
    try:
        return ENV_REGISTRY[env_type]
    except KeyError:
        available_types = list(ENV_REGISTRY.keys())
        raise ValueError(
            f"Unsupported environment type: '{env_type}'. Available types: {available_types}"
        )


def list_supported_env_types() -> List[str]:
    """
    List all registered environment type names.

    This function returns a list of all environment names that have been
    registered in the global registry. It's useful for:
    - Displaying available options to users
    - Validating environment type parameters
    - Building dynamic UIs or configuration tools
    - Debugging and development

    Returns:
        List[str]: Sorted list of all registered environment type names.
            Returns an empty list if no environments have been registered.

    Example:
        >>> # Check what environments are available
        >>> available_envs = list_supported_env_types()
        >>> print("Supported environments:")
        >>> for env_type in available_envs:
        ...     print(f"  - {env_type}")

        >>> # Validate user input
        >>> user_choice = input("Choose environment: ")
        >>> if user_choice not in list_supported_env_types():
        ...     print(f"Error: {user_choice} not available")

        >>> # Build configuration options
        >>> config_schema = {
        ...     "environment_type": {
        ...         "enum": list_supported_env_types(),
        ...         "description": "Type of environment to use"
        ...     }
        ... }

    Note:
        The returned list is sorted alphabetically for consistent ordering.
        This function returns a copy of the environment names, so modifying
        the returned list will not affect the registry.
    """
    return sorted(list(ENV_REGISTRY.keys()))


def discover_entry_point_environments() -> None:
    """
    Discover and register environments from entry points.
    
    This function scans for environments registered via setuptools entry points
    under the group 'synth_ai.environments'. This allows third-party packages
    to register environments by declaring them in their pyproject.toml:
    
    [project.entry-points."synth_ai.environments"]
    my_env = "my_package.my_env:MyEnvironment"
    another_env = "my_package.other:AnotherEnv"
    
    The function will automatically import and register all discovered environments.
    """
    try:
        entry_points = importlib.metadata.entry_points()
        if hasattr(entry_points, 'select'):
            # Python 3.10+
            env_entry_points = entry_points.select(group='synth_ai.environments')
        else:
            # Python 3.9 and below
            env_entry_points = entry_points.get('synth_ai.environments', [])
        
        for entry_point in env_entry_points:
            try:
                env_cls = entry_point.load()
                
                # Validate that it's a StatefulEnvironment subclass
                if not issubclass(env_cls, StatefulEnvironment):
                    logger.warning(
                        f"Entry point '{entry_point.name}' from '{entry_point.value}' "
                        f"is not a StatefulEnvironment subclass. Skipping."
                    )
                    continue
                
                register_environment(entry_point.name, env_cls)
                logger.info(f"Registered environment '{entry_point.name}' from entry point")
                
            except Exception as e:
                logger.error(
                    f"Failed to load environment entry point '{entry_point.name}' "
                    f"from '{entry_point.value}': {e}"
                )
    except Exception as e:
        logger.debug(f"Entry point discovery failed (this is normal if no entry points exist): {e}")


def auto_discover_environments() -> None:
    """
    Automatically discover and register environments from multiple sources.
    
    This function combines multiple discovery mechanisms:
    1. Entry points (setuptools plugins)
    2. Could be extended with more discovery methods in the future
    
    This should be called once at application startup to populate the registry
    with all available environments.
    """
    discover_entry_point_environments()


# Auto-discover environments when the registry module is imported
# This ensures third-party environments are available as soon as the registry is used
auto_discover_environments()
