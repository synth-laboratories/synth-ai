"""Public Managed Research actor-type/subtype model policy helpers.

Source of truth: backend/config/smr_actor_model_policy.json
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from synth_ai.managed_research.models.smr_actor_policy_data import (
    SMR_ACTOR_MODEL_POLICY,
    SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES,
)
from synth_ai.managed_research.models.smr_agent_models import SmrAgentModel, coerce_smr_agent_model


class SmrActorType(StrEnum):
    ORCHESTRATOR = "orchestrator"
    REVIEWER = "reviewer"
    WORKER = "worker"


class SmrActorSubtype(StrEnum):
    MAIN = "main"
    ENGINEER = "engineer"
    RESEARCHER = "researcher"
    ARTIFACT_BUILDER = "artifact_builder"
    ARTIFACT_REVIEWER = "artifact_reviewer"
    TASK_COMPLETION = "task_completion"
    RUN_COMPLETION = "run_completion"
    SAFETY = "safety"
    OBJECTIVE = "objective"


class SmrOrchestratorSubtype(StrEnum):
    MAIN = "main"


class SmrReviewerSubtype(StrEnum):
    MAIN = "main"
    TASK_COMPLETION = "task_completion"
    RUN_COMPLETION = "run_completion"
    SAFETY = "safety"
    OBJECTIVE = "objective"
    ARTIFACT_REVIEWER = "artifact_reviewer"


class SmrWorkerSubtype(StrEnum):
    ENGINEER = "engineer"
    RESEARCHER = "researcher"
    ARTIFACT_BUILDER = "artifact_builder"


SMR_ACTOR_TYPE_VALUES: tuple[str, ...] = tuple(item.value for item in SmrActorType)
SMR_ACTOR_SUBTYPE_VALUES: tuple[str, ...] = tuple(item.value for item in SmrActorSubtype)
SMR_ORCHESTRATOR_SUBTYPE_VALUES: tuple[str, ...] = tuple(
    item.value for item in SmrOrchestratorSubtype
)
SMR_REVIEWER_SUBTYPE_VALUES: tuple[str, ...] = tuple(item.value for item in SmrReviewerSubtype)
SMR_WORKER_SUBTYPE_VALUES: tuple[str, ...] = tuple(item.value for item in SmrWorkerSubtype)
SMR_ACTOR_SUBTYPE_VALUES_BY_TYPE: dict[str, tuple[str, ...]] = {
    SmrActorType.ORCHESTRATOR.value: SMR_ORCHESTRATOR_SUBTYPE_VALUES,
    SmrActorType.REVIEWER.value: SMR_REVIEWER_SUBTYPE_VALUES,
    SmrActorType.WORKER.value: SMR_WORKER_SUBTYPE_VALUES,
}


@dataclass(frozen=True, slots=True)
class SmrActorModelAssignment:
    actor_type: SmrActorType
    actor_subtype: SmrActorSubtype
    agent_model: SmrAgentModel
    agent_model_params: dict[str, Any] | None = None
    agent_harness: str | None = None
    provider: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "actor_type": self.actor_type.value,
            "actor_subtype": self.actor_subtype.value,
            "agent_model": self.agent_model.value,
        }
        if self.agent_model_params:
            payload["agent_model_params"] = dict(self.agent_model_params)
        if self.agent_harness:
            payload["agent_harness"] = self.agent_harness
            payload["agent_kind"] = self.agent_harness
        if self.provider:
            payload["provider"] = dict(self.provider)
        return payload


def coerce_smr_actor_type(
    value: SmrActorType | str | None,
    *,
    field_name: str = "actor_type",
) -> SmrActorType:
    if isinstance(value, SmrActorType):
        return value
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    try:
        return SmrActorType(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_ACTOR_TYPE_VALUES)}"
        ) from exc


def coerce_smr_actor_subtype(
    value: SmrActorSubtype | str | None,
    *,
    actor_type: SmrActorType | str,
    field_name: str = "actor_subtype",
) -> SmrActorSubtype:
    resolved_actor_type = coerce_smr_actor_type(actor_type, field_name="actor_type")
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    try:
        subtype = SmrActorSubtype(normalized)
    except ValueError as exc:
        allowed = SMR_ACTOR_SUBTYPE_VALUES_BY_TYPE[resolved_actor_type.value]
        raise ValueError(f"{field_name} must be one of: {', '.join(allowed)}") from exc
    if subtype.value not in SMR_ACTOR_SUBTYPE_VALUES_BY_TYPE[resolved_actor_type.value]:
        raise ValueError(
            f"{field_name} '{subtype.value}' is not valid for actor_type '{resolved_actor_type.value}'"
        )
    return subtype


def _permitted_models_for_actor(
    actor_type: SmrActorType, actor_subtype: SmrActorSubtype
) -> tuple[str, ...]:
    for entry in SMR_ACTOR_MODEL_POLICY:
        if (
            entry["actor_type"] == actor_type.value
            and entry["actor_subtype"] == actor_subtype.value
        ):
            return tuple(str(item) for item in entry.get("permitted_models") or ())
    raise ValueError(
        f"No Managed Research actor model policy entry exists for {actor_type.value}:{actor_subtype.value}"
    )


def validate_shared_top_level_agent_model(
    value: SmrAgentModel | str | None,
    *,
    field_name: str = "agent_model",
) -> SmrAgentModel | None:
    model = coerce_smr_agent_model(value, field_name=field_name)
    if model is None:
        return None
    if model.value not in SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES:
        raise ValueError(
            f"{field_name} '{model.value}' is not allowed for shared top-level selection; allowed values are: {', '.join(SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES)}"
        )
    return model


def coerce_smr_actor_model_assignment(
    value: SmrActorModelAssignment | Mapping[str, Any],
    *,
    field_name: str = "actor_model_assignments",
) -> SmrActorModelAssignment:
    if isinstance(value, SmrActorModelAssignment):
        assignment = value
    else:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} entries must be objects")
        actor_type = coerce_smr_actor_type(
            value.get("actor_type"),
            field_name=f"{field_name}.actor_type",
        )
        actor_subtype = coerce_smr_actor_subtype(
            value.get("actor_subtype"),
            actor_type=actor_type,
            field_name=f"{field_name}.actor_subtype",
        )
        agent_model = coerce_smr_agent_model(
            value.get("agent_model"),
            field_name=f"{field_name}.agent_model",
        )
        if agent_model is None:
            raise ValueError(f"{field_name}.agent_model is required")
        params = value.get("agent_model_params")
        if params is not None and not isinstance(params, Mapping):
            raise ValueError(f"{field_name}.agent_model_params must be an object when provided")
        agent_harness = (
            str(value.get("agent_harness") or value.get("agent_kind") or "").strip() or None
        )
        provider_raw = value.get("provider")
        if provider_raw is None:
            provider_id = str(value.get("provider_id") or "").strip()
            provider = {"provider_id": provider_id} if provider_id else None
        elif isinstance(provider_raw, str):
            provider_id = provider_raw.strip()
            provider = {"provider_id": provider_id} if provider_id else None
        elif isinstance(provider_raw, Mapping):
            provider = {str(key): item for key, item in provider_raw.items()}
        else:
            raise ValueError(f"{field_name}.provider must be an object or string when provided")
        assignment = SmrActorModelAssignment(
            actor_type=actor_type,
            actor_subtype=actor_subtype,
            agent_model=agent_model,
            agent_model_params=dict(params) if isinstance(params, Mapping) else None,
            agent_harness=agent_harness,
            provider=provider,
        )
    permitted = _permitted_models_for_actor(assignment.actor_type, assignment.actor_subtype)
    if assignment.agent_model.value not in permitted:
        raise ValueError(
            f"{field_name}.agent_model '{assignment.agent_model.value}' is not permitted for {assignment.actor_type.value}:{assignment.actor_subtype.value}; allowed values are: {', '.join(permitted)}"
        )
    return assignment


def normalize_actor_model_assignments(
    value: Any,
    *,
    field_name: str,
) -> list[SmrActorModelAssignment]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array of objects")
    assignments: list[SmrActorModelAssignment] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        assignment = coerce_smr_actor_model_assignment(
            item,
            field_name=f"{field_name}[{index}]",
        )
        key = (assignment.actor_type.value, assignment.actor_subtype.value)
        if key in seen:
            raise ValueError(
                f"{field_name} contains duplicate entry for {assignment.actor_type.value}:{assignment.actor_subtype.value}"
            )
        seen.add(key)
        assignments.append(assignment)
    return assignments


__all__ = [
    "SMR_ACTOR_MODEL_POLICY",
    "SMR_ACTOR_SUBTYPE_VALUES",
    "SMR_ACTOR_SUBTYPE_VALUES_BY_TYPE",
    "SMR_ACTOR_TYPE_VALUES",
    "SMR_ORCHESTRATOR_SUBTYPE_VALUES",
    "SMR_REVIEWER_SUBTYPE_VALUES",
    "SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES",
    "SMR_WORKER_SUBTYPE_VALUES",
    "SmrActorModelAssignment",
    "SmrActorSubtype",
    "SmrActorType",
    "SmrOrchestratorSubtype",
    "SmrReviewerSubtype",
    "SmrWorkerSubtype",
    "coerce_smr_actor_model_assignment",
    "coerce_smr_actor_subtype",
    "coerce_smr_actor_type",
    "normalize_actor_model_assignments",
    "validate_shared_top_level_agent_model",
]
