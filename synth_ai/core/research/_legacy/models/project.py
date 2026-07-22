"""Typed public project models mirrored from the backend contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from synth_ai.core.research._legacy.models.run_state import (
    _optional_bool,
    _optional_object_dict,
    _optional_string,
    _require_mapping,
    _require_string,
)


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    raise ValueError(f"{key} must be null, a datetime, or an ISO-8601 string")


@dataclass(frozen=True)
class ManagedResearchProject:
    project_id: str
    org_id: str
    name: str
    project_alias: str | None = None
    project_kind: str | None = None
    timezone: str | None = None
    schedule: dict[str, object] = field(default_factory=dict)
    budgets: dict[str, object] = field(default_factory=dict)
    key_policy: dict[str, object] = field(default_factory=dict)
    integrations: dict[str, object] = field(default_factory=dict)
    project_repo: dict[str, object] = field(default_factory=dict)
    repos: list[str] = field(default_factory=list)
    onboarding_state: dict[str, object] = field(default_factory=dict)
    research: dict[str, object] = field(default_factory=dict)
    source_repo: dict[str, object] | None = None
    synth_ai: dict[str, object] = field(default_factory=dict)
    policy: dict[str, object] = field(default_factory=dict)
    trial_matrix: dict[str, object] = field(default_factory=dict)
    execution: dict[str, object] = field(default_factory=dict)
    active_config_version_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    archived: bool = False
    archived_at: datetime | None = None
    active_run_id: str | None = None
    latest_run_id: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchProject:
        mapping = _require_mapping(payload, label="managed research project")
        repos_payload = mapping.get("repos")
        repos = (
            [str(item) for item in repos_payload if isinstance(item, str)]
            if isinstance(repos_payload, list)
            else []
        )
        source_repo_payload = mapping.get("source_repo")
        return cls(
            project_id=_require_string(mapping, "project_id", label="project.project_id"),
            project_alias=_optional_string(mapping, "project_alias"),
            project_kind=_optional_string(mapping, "project_kind"),
            org_id=_require_string(mapping, "org_id", label="project.org_id"),
            name=_require_string(mapping, "name", label="project.name"),
            timezone=_optional_string(mapping, "timezone"),
            schedule=_optional_object_dict(mapping.get("schedule"), label="project.schedule"),
            budgets=_optional_object_dict(mapping.get("budgets"), label="project.budgets"),
            key_policy=_optional_object_dict(
                mapping.get("key_policy"),
                label="project.key_policy",
            ),
            integrations=_optional_object_dict(
                mapping.get("integrations"),
                label="project.integrations",
            ),
            project_repo=_optional_object_dict(
                mapping.get("project_repo"),
                label="project.project_repo",
            ),
            repos=repos,
            onboarding_state=_optional_object_dict(
                mapping.get("onboarding_state"),
                label="project.onboarding_state",
            ),
            research=_optional_object_dict(mapping.get("research"), label="project.research"),
            source_repo=(
                _optional_object_dict(source_repo_payload, label="project.source_repo")
                if source_repo_payload is not None
                else None
            ),
            synth_ai=_optional_object_dict(mapping.get("synth_ai"), label="project.synth_ai"),
            policy=_optional_object_dict(mapping.get("policy"), label="project.policy"),
            trial_matrix=_optional_object_dict(
                mapping.get("trial_matrix"),
                label="project.trial_matrix",
            ),
            execution=_optional_object_dict(mapping.get("execution"), label="project.execution"),
            active_config_version_id=_optional_string(mapping, "active_config_version_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            archived=bool(_optional_bool(mapping, "archived")),
            archived_at=_optional_datetime(mapping, "archived_at"),
            active_run_id=_optional_string(mapping, "active_run_id"),
            latest_run_id=_optional_string(mapping, "latest_run_id"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class CreateRunnableResult:
    project_id: str
    name: str
    project: ManagedResearchProject
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> CreateRunnableResult:
        project = ManagedResearchProject.from_wire(payload)
        return cls(
            project_id=project.project_id,
            name=project.name,
            project=project,
            raw=dict(project.raw),
        )

    def __getitem__(self, key: str) -> object:
        return self.raw[key]

    def get(self, key: str, default: object = None) -> object:
        return self.raw.get(key, default)


__all__ = ["CreateRunnableResult", "ManagedResearchProject"]
