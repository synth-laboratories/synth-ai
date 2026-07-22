"""Opaque identity types for Research resources."""

from __future__ import annotations


class ProjectId(str):
    """Backend-owned Research project identifier."""


class SwarmId(str):
    """Backend-owned Research swarm identifier."""


class ActorId(str):
    """Backend-owned actor identifier within a Research swarm."""


class TaskId(str):
    """Backend-owned task identifier within a Research swarm."""


class MessageId(str):
    """Backend-owned message identifier within a Research swarm."""


class ActivityEventId(str):
    """Backend-owned activity-event identifier within a Research swarm."""


class ArtifactId(str):
    """Backend-owned durable Research artifact identifier."""


class WorkProductId(str):
    """Backend-owned durable Research WorkProduct identifier."""


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
    "ActivityEventId",
    "ActorId",
    "ArtifactId",
    "ConfigurationVersionId",
    "EffortId",
    "EnvironmentKind",
    "FactoryId",
    "MessageId",
    "OrganizationId",
    "PoolId",
    "ProfileId",
    "ProjectId",
    "RuntimeKind",
    "SwarmId",
    "TaskId",
    "WorkProductId",
]
