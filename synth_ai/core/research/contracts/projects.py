"""Typed project contracts from the bounded backend Research schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    object_value,
    optional_bool,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    EnvironmentKind,
    OrganizationId,
    PoolId,
    ProfileId,
    ProjectId,
    RuntimeKind,
    SwarmId,
    require_text,
)


class ProjectSetupState(StrEnum):
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    BLOCKED = "blocked"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    name: str
    pool_id: PoolId
    runtime_kind: RuntimeKind
    environment_kind: EnvironmentKind
    orchestrator_profile_id: ProfileId
    default_worker_profile_id: ProfileId
    worker_profile_ids: tuple[ProfileId, ...] = ()
    actor_profile_id: ProfileId | None = None
    runtime_artifact_release_id: str | None = None
    timezone: str = "UTC"
    scenario: str | None = None
    notes: str | None = None
    execution_policy: JsonObject = field(default_factory=dict)
    research: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("name", self.name),
            ("pool_id", self.pool_id),
            ("runtime_kind", self.runtime_kind),
            ("environment_kind", self.environment_kind),
            ("orchestrator_profile_id", self.orchestrator_profile_id),
            ("default_worker_profile_id", self.default_worker_profile_id),
            ("timezone", self.timezone),
        ):
            require_text(value, field_name=name)

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "pool_id": self.pool_id,
            "runtime_kind": self.runtime_kind,
            "environment_kind": self.environment_kind,
            "orchestrator_profile_id": self.orchestrator_profile_id,
            "default_worker_profile_id": self.default_worker_profile_id,
            "worker_profile_ids": list(self.worker_profile_ids),
            "timezone": self.timezone,
            "execution_policy": dict(self.execution_policy),
            "research": dict(self.research),
        }
        for name, value in (
            ("actor_profile_id", self.actor_profile_id),
            ("runtime_artifact_release_id", self.runtime_artifact_release_id),
            ("scenario", self.scenario),
            ("notes", self.notes),
        ):
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True, slots=True)
class ProjectPatch:
    name: str | None = None
    timezone: str | None = None
    archived: bool | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            require_text(self.name, field_name="name")
        if self.timezone is not None:
            require_text(self.timezone, field_name="timezone")
        if self.name is None and self.timezone is None and self.archived is None:
            raise ValueError("project patch must include at least one field")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {}
        if self.name is not None:
            payload["name"] = self.name
        if self.timezone is not None:
            payload["timezone"] = self.timezone
        if self.archived is not None:
            payload["archived"] = self.archived
        return payload


@dataclass(frozen=True, slots=True)
class Project:
    project_id: ProjectId
    organization_id: OrganizationId
    name: str
    timezone: str
    created_at: datetime
    updated_at: datetime
    archived: bool = False
    project_alias: str | None = None
    project_kind: str | None = None
    active_swarm_id: SwarmId | None = None
    latest_swarm_id: SwarmId | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> Project:
        payload = object_value(value, operation_id="project")
        active_swarm = optional_text(payload, "active_run_id")
        latest_swarm = optional_text(payload, "latest_run_id")
        return cls(
            project_id=ProjectId(required_text(payload, "project_id")),
            organization_id=OrganizationId(required_text(payload, "org_id")),
            name=required_text(payload, "name"),
            timezone=required_text(payload, "timezone"),
            created_at=required_datetime(payload, "created_at"),
            updated_at=required_datetime(payload, "updated_at"),
            archived=optional_bool(payload, "archived"),
            project_alias=optional_text(payload, "project_alias"),
            project_kind=optional_text(payload, "project_kind"),
            active_swarm_id=SwarmId(active_swarm) if active_swarm is not None else None,
            latest_swarm_id=SwarmId(latest_swarm) if latest_swarm is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ProjectSetup:
    project_id: ProjectId
    state: ProjectSetupState
    blockers: tuple[str, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectSetup:
        payload = object_value(value, operation_id="project setup")
        blockers_value = payload.get("blockers", [])
        if not isinstance(blockers_value, list) or not all(
            isinstance(blocker, str) for blocker in blockers_value
        ):
            raise ValueError("project setup blockers must be an array of strings")
        return cls(
            ProjectId(required_text(payload, "project_id")),
            ProjectSetupState(required_text(payload, "state")),
            tuple(blockers_value),
        )


ResearchProject = Project
ResearchProjectCreateRequest = ProjectSpec
ResearchProjectPatchRequest = ProjectPatch
ResearchProjectSetup = ProjectSetup


__all__ = [
    "ProjectSetupState",
    "Project",
    "ProjectSpec",
    "ProjectPatch",
    "ProjectSetup",
    "ResearchProject",
    "ResearchProjectCreateRequest",
    "ResearchProjectPatchRequest",
    "ResearchProjectSetup",
]
