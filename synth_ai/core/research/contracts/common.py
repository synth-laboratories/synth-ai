"""Opaque identity types for Research resources."""

from __future__ import annotations


class ProjectId(str):
    """Backend-owned Research project identifier."""


class SwarmId(str):
    """Backend-owned Research swarm identifier."""


class ConfigurationVersionId(str):
    """Immutable backend configuration-snapshot identifier."""


class EffortId(str):
    """Backend-owned Factory Effort identifier."""


class FactoryId(str):
    """Backend-owned Research Factory identifier."""


class OrganizationId(str):
    """Backend-owned organization identifier."""


class PoolId(str):
    """Backend-owned compute pool identifier."""


class ProfileId(str):
    """Backend-owned actor profile identifier."""


class RuntimeKind(str):
    """Backend catalog runtime-kind identifier."""


class EnvironmentKind(str):
    """Backend catalog environment-kind identifier."""


def require_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


__all__ = [
    "ConfigurationVersionId",
    "EffortId",
    "EnvironmentKind",
    "FactoryId",
    "OrganizationId",
    "PoolId",
    "ProfileId",
    "ProjectId",
    "RuntimeKind",
    "SwarmId",
]
