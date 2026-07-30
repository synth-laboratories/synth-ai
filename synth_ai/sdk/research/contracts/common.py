"""Opaque identity types for Research resources."""

from __future__ import annotations


class ProjectId(str):
    """Backend-owned Research project identifier."""


class ProjectRepositoryId(str):
    """Backend-owned external repository attached to a Research project."""


class ProjectDatasetId(str):
    """Backend-owned dataset attached to a Research project."""


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


class TranscriptEventId(str):
    """Backend-owned transcript-event identifier within a Research swarm."""


class ParticipantSessionId(str):
    """Backend-owned participant session represented in a transcript."""


class ArtifactId(str):
    """Backend-owned durable Research artifact identifier."""


class WorkProductId(str):
    """Backend-owned durable Research WorkProduct identifier."""


class ConfigurationVersionId(str):
    """Immutable backend configuration-snapshot identifier."""


class WorkspaceFileId(str):
    """Backend-owned stored workspace-file identifier."""


class ProjectEventId(str):
    """Backend-owned project collaboration-event identifier."""


class EffortId(str):
    """Backend-owned Factory Effort identifier."""


class FactoryId(str):
    """Backend-owned Research Factory identifier."""


class FactoryCandidateId(str):
    """Backend-owned immutable Factory candidate identifier."""


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


class EnvironmentId(str):
    """Backend-owned immutable Environment catalog identifier."""


class EnvironmentName(str):
    """Human-readable Environment catalog selector."""


class EnvironmentDigest(str):
    """Content-addressed Environment manifest digest."""


class UserId(str):
    """Backend-owned user identifier."""


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
    "EnvironmentDigest",
    "EnvironmentId",
    "EnvironmentKind",
    "EnvironmentName",
    "FactoryCandidateId",
    "FactoryId",
    "MessageId",
    "OrganizationId",
    "ParticipantSessionId",
    "PoolId",
    "ProfileId",
    "ProjectDatasetId",
    "ProjectEventId",
    "ProjectId",
    "ProjectRepositoryId",
    "RuntimeKind",
    "SwarmId",
    "TaskId",
    "TranscriptEventId",
    "UserId",
    "WorkspaceFileId",
    "WorkProductId",
]
