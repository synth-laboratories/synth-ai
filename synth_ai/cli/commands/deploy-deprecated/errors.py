from __future__ import annotations

from dataclasses import dataclass


class DeployCliError(RuntimeError):
    """Base exception for deploy CLI failures."""


@dataclass(slots=True)
class MissingEnvironmentApiKeyError(DeployCliError):
    """Raised when ENVIRONMENT_API_KEY is absent and cannot be collected interactively."""

    hint: str | None = None


@dataclass(slots=True)
class EnvironmentKeyLoadError(DeployCliError):
    """Raised when we fail to persist or reload ENVIRONMENT_API_KEY from disk."""

    path: str | None = None


@dataclass(slots=True)
class EnvFileDiscoveryError(DeployCliError):
    """Raised when no suitable env file can be found for a task app."""

    attempted: tuple[str, ...] = ()
    hint: str | None = None


@dataclass(slots=True)
class TaskAppNotFoundError(DeployCliError):
    """Raised when the requested task app identifier cannot be resolved."""

    app_id: str | None = None
    available: tuple[str, ...] = ()


@dataclass(slots=True)
class ModalCliResolutionError(DeployCliError):
    """Raised when the Modal CLI executable cannot be located or invoked."""

    cli_path: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class ModalExecutionError(DeployCliError):
    """Raised when a Modal subprocess exits with a non-zero status."""

    command: str
    exit_code: int


@dataclass(slots=True)
class EnvKeyPreflightError(DeployCliError):
    """Raised when uploading or minting ENVIRONMENT_API_KEY to the backend fails."""

    detail: str | None = None


__all__ = [
    "DeployCliError",
    "MissingEnvironmentApiKeyError",
    "EnvironmentKeyLoadError",
    "EnvFileDiscoveryError",
    "TaskAppNotFoundError",
    "ModalCliResolutionError",
    "ModalExecutionError",
    "EnvKeyPreflightError",
]
