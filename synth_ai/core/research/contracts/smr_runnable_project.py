"""Typed runnable-project create request contract.

# See: backend SMR runnable project create schemas (authority).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from synth_ai.core.research.contracts.smr_actor_models import (
    SmrActorModelAssignment,
    normalize_actor_model_assignments,
)
from synth_ai.core.research.contracts.smr_environment_kinds import (
    SmrEnvironmentKind,
    coerce_smr_environment_kind,
)
from synth_ai.core.research.contracts.smr_runtime_kinds import (
    SmrRuntimeKind,
    coerce_smr_runtime_kind,
)


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], payload)


def _optional_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _require_string(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _object_dict(payload: object) -> dict[str, object]:
    mapping = _require_mapping(payload, label="metadata")
    return dict(mapping)


def _optional_object_dict(payload: object) -> dict[str, object]:
    if payload is None:
        return {}
    return _object_dict(payload)


def _string_list(payload: object, *, label: str) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    values: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        normalized = item.strip()
        if normalized:
            values.append(normalized)
    return values


@dataclass(frozen=True)
class SmrAgentProfileBindings:
    orchestrator_profile_id: str
    default_worker_profile_id: str
    worker_profile_ids: list[str] = field(default_factory=list)

    def to_wire(self) -> dict[str, object]:
        worker_profile_ids = list(self.worker_profile_ids)
        if not worker_profile_ids:
            worker_profile_ids = [self.default_worker_profile_id]
        return {
            "orchestrator_profile_id": self.orchestrator_profile_id,
            "default_worker_profile_id": self.default_worker_profile_id,
            "worker_profile_ids": worker_profile_ids,
        }


@dataclass(frozen=True)
class SmrRunnableProjectRequest:
    name: str
    timezone: str
    pool_id: str
    runtime_kind: SmrRuntimeKind
    environment_kind: SmrEnvironmentKind
    agent_profiles: SmrAgentProfileBindings
    worker_profile_ids: list[str] = field(default_factory=list)
    actor_profile_id: str | None = None
    actor_model_assignments: list[SmrActorModelAssignment] = field(default_factory=list)
    runtime_artifact_release_id: str | None = None
    budgets: dict[str, object] = field(default_factory=dict)
    key_policy: dict[str, object] = field(default_factory=dict)
    execution_policy: dict[str, object] = field(default_factory=dict)
    research: dict[str, object] = field(default_factory=dict)
    scenario: str | None = None
    notes: str | None = None
    retention_policy: dict[str, object] = field(default_factory=dict)
    metered_infra: dict[str, object] = field(default_factory=dict)
    schedule: dict[str, object] = field(default_factory=dict)
    integrations: dict[str, object] = field(default_factory=dict)
    synth_ai: dict[str, object] = field(default_factory=dict)
    policy: dict[str, object] = field(default_factory=dict)
    trial_matrix: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunnableProjectRequest:
        mapping = _require_mapping(payload, label="runnable project request")
        agent_profiles_payload = _require_mapping(
            mapping.get("agent_profiles"),
            label="runnable project request.agent_profiles",
        )
        worker_profile_ids = _string_list(
            agent_profiles_payload.get("worker_profile_ids"),
            label="runnable project request.agent_profiles.worker_profile_ids",
        )
        return cls(
            name=_require_string(mapping, "name", label="runnable project request.name"),
            timezone=_require_string(
                mapping,
                "timezone",
                label="runnable project request.timezone",
            ),
            pool_id=_require_string(
                mapping,
                "pool_id",
                label="runnable project request.pool_id",
            ),
            runtime_kind=coerce_smr_runtime_kind(
                _require_string(
                    mapping,
                    "runtime_kind",
                    label="runnable project request.runtime_kind",
                ),
                field_name="runtime_kind",
            )
            or SmrRuntimeKind.SANDBOX_AGENT,
            environment_kind=coerce_smr_environment_kind(
                _require_string(
                    mapping,
                    "environment_kind",
                    label="runnable project request.environment_kind",
                ),
                field_name="environment_kind",
            )
            or SmrEnvironmentKind.HARBOR,
            agent_profiles=SmrAgentProfileBindings(
                orchestrator_profile_id=_require_string(
                    agent_profiles_payload,
                    "orchestrator_profile_id",
                    label="runnable project request.agent_profiles.orchestrator_profile_id",
                ),
                default_worker_profile_id=_require_string(
                    agent_profiles_payload,
                    "default_worker_profile_id",
                    label="runnable project request.agent_profiles.default_worker_profile_id",
                ),
                worker_profile_ids=worker_profile_ids,
            ),
            worker_profile_ids=worker_profile_ids,
            actor_profile_id=_optional_string(mapping, "actor_profile_id"),
            actor_model_assignments=normalize_actor_model_assignments(
                mapping.get("actor_model_assignments"),
                field_name="actor_model_assignments",
            ),
            runtime_artifact_release_id=_optional_string(mapping, "runtime_artifact_release_id"),
            budgets=_optional_object_dict(mapping.get("budgets")),
            key_policy=_optional_object_dict(mapping.get("key_policy")),
            execution_policy=_optional_object_dict(mapping.get("execution_policy")),
            research=_optional_object_dict(mapping.get("research")),
            scenario=_optional_string(mapping, "scenario"),
            notes=_optional_string(mapping, "notes"),
            retention_policy=_optional_object_dict(mapping.get("retention_policy")),
            metered_infra=_optional_object_dict(mapping.get("metered_infra")),
            schedule=_optional_object_dict(mapping.get("schedule")),
            integrations=_optional_object_dict(mapping.get("integrations")),
            synth_ai=_optional_object_dict(mapping.get("synth_ai")),
            policy=_optional_object_dict(mapping.get("policy")),
            trial_matrix=_optional_object_dict(mapping.get("trial_matrix")),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "timezone": self.timezone,
            "pool_id": self.pool_id,
            "runtime_kind": self.runtime_kind.value,
            "environment_kind": self.environment_kind.value,
            "orchestrator_profile_id": self.agent_profiles.orchestrator_profile_id,
            "default_worker_profile_id": self.agent_profiles.default_worker_profile_id,
            "worker_profile_ids": list(
                self.agent_profiles.worker_profile_ids or self.worker_profile_ids
            ),
            "budgets": dict(self.budgets),
            "key_policy": dict(self.key_policy),
            "execution_policy": dict(self.execution_policy),
            "research": dict(self.research),
            "retention_policy": dict(self.retention_policy),
            "metered_infra": dict(self.metered_infra),
            "schedule": dict(self.schedule),
            "integrations": dict(self.integrations),
            "synth_ai": dict(self.synth_ai),
            "policy": dict(self.policy),
            "trial_matrix": dict(self.trial_matrix),
        }
        if self.actor_model_assignments:
            payload["actor_model_assignments"] = [
                item.as_payload() for item in self.actor_model_assignments
            ]
        if self.actor_profile_id is not None:
            payload["actor_profile_id"] = self.actor_profile_id
        if self.runtime_artifact_release_id is not None:
            payload["runtime_artifact_release_id"] = self.runtime_artifact_release_id
        if self.scenario is not None:
            payload["scenario"] = self.scenario
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


__all__ = [
    "SmrAgentProfileBindings",
    "SmrRunnableProjectRequest",
]
