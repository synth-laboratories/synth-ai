"""
External environment registry support.

This module provides functionality to register environments from external packages.
"""

import importlib
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ExternalRegistryConfig:
    """Configuration for external environment registries."""

    def __init__(self, external_environments: List[Dict[str, str]] = None):
        self.external_environments = external_environments or []


def load_external_environments(config: ExternalRegistryConfig):
    """
    Load and register environments from external packages.

    Args:
        config: Configuration specifying external environment sources
    """
    for env_config in config.external_environments:
        module_name = env_config.get("module")
        function_name = env_config.get("function", "integrate_with_environments_service")

        if not module_name:
            logger.warning("External environment config missing 'module' field")
            continue

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Get the registration function
            if hasattr(module, function_name):
                register_func = getattr(module, function_name)
                register_func()
                logger.info(f"Successfully loaded environments from {module_name}")
            else:
                logger.warning(f"Module {module_name} does not have function {function_name}")

        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading environments from {module_name}: {e}")


__all__ = [
    "ExternalRegistryConfig",
    "load_external_environments",
]
