from __future__ import annotations

from .core import command, get_command
from .errors import (
    DeployCliError,
    EnvFileDiscoveryError,
    EnvironmentKeyLoadError,
    EnvKeyPreflightError,
    MissingEnvironmentApiKeyError,
    ModalCliResolutionError,
    ModalExecutionError,
    TaskAppNotFoundError,
)
from .validation import validate_deploy_options

# Alias for backwards compatibility
deploy_cmd = command

__all__ = [
    "command",
    "get_command",
    "deploy_cmd",
    "DeployCliError",
    "MissingEnvironmentApiKeyError",
    "EnvironmentKeyLoadError",
    "EnvFileDiscoveryError",
    "TaskAppNotFoundError",
    "ModalCliResolutionError",
    "ModalExecutionError",
    "EnvKeyPreflightError",
    "validate_deploy_options",
]
