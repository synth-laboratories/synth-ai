"""Typed role-based launch policy models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from synth_ai.core.research._legacy.models.smr_actor_models import SmrWorkerSubtype
from synth_ai.core.research._legacy.models.smr_agent_harnesses import (
    SmrAgentHarness,
    coerce_smr_agent_harness,
)
from synth_ai.core.research._legacy.models.smr_agent_models import (
    SmrAgentModel,
    coerce_smr_agent_model,
)
from synth_ai.core.research._legacy.models.smr_providers import Provider, coerce_provider


def _require_mapping(payload: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return payload


def _optional_mapping(payload: object, *, field_name: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return {str(key): value for key, value in payload.items()}


def _coerce_worker_subtype(
    value: SmrWorkerSubtype | str,
    *,
    field_name: str,
) -> SmrWorkerSubtype:
    if isinstance(value, SmrWorkerSubtype):
        return value
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    try:
        return SmrWorkerSubtype(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SmrWorkerSubtype)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class RoleProviderRequirement:
    provider_id: Provider
    provider_required: bool = True
    provider_model_id: str | None = None
    route_alias: str | None = None

    @classmethod
    def from_wire(
        cls,
        payload: RoleProviderRequirement | Provider | str | Mapping[str, Any] | None,
        *,
        field_name: str = "roles.binding.provider",
    ) -> RoleProviderRequirement | None:
        if payload is None:
            return None
        if isinstance(payload, RoleProviderRequirement):
            return payload
        if isinstance(payload, (Provider, str)):
            provider = coerce_provider(payload, field_name=field_name)
            if provider is None:
                return None
            return cls(provider_id=provider)
        mapping = _require_mapping(payload, field_name=field_name)
        provider = coerce_provider(
            mapping.get("provider_id") or mapping.get("provider") or mapping.get("kind"),
            field_name=f"{field_name}.provider_id",
        )
        if provider is None:
            raise ValueError(f"{field_name}.provider_id is required")
        return cls(
            provider_id=provider,
            provider_required=bool(mapping.get("provider_required", True)),
            provider_model_id=str(mapping.get("provider_model_id") or "").strip() or None,
            route_alias=str(mapping.get("route_alias") or "").strip() or None,
        )

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_id": self.provider_id.value,
            "provider_required": bool(self.provider_required),
        }
        if self.provider_model_id:
            payload["provider_model_id"] = self.provider_model_id
        if self.route_alias:
            payload["route_alias"] = self.route_alias
        return payload


@dataclass(frozen=True, slots=True)
class RoleBinding:
    model: SmrAgentModel
    params: dict[str, Any] = field(default_factory=dict)
    agent_harness: SmrAgentHarness | None = None
    provider: RoleProviderRequirement | Provider | str | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized_params = _optional_mapping(
            self.params,
            field_name="roles.binding.params",
        )
        object.__setattr__(self, "params", normalized_params or {})
        resolved_harness = coerce_smr_agent_harness(
            self.agent_harness,
            field_name="roles.binding.agent_harness",
        )
        object.__setattr__(self, "agent_harness", resolved_harness)
        object.__setattr__(
            self,
            "provider",
            RoleProviderRequirement.from_wire(
                self.provider,
                field_name="roles.binding.provider",
            ),
        )

    @classmethod
    def from_wire(cls, payload: object) -> RoleBinding:
        mapping = _require_mapping(payload, field_name="roles.binding")
        model = coerce_smr_agent_model(
            mapping.get("model"),
            field_name="roles.binding.model",
        )
        if model is None:
            raise ValueError("roles.binding.model is required")
        params = _optional_mapping(
            mapping.get("params"),
            field_name="roles.binding.params",
        )
        harness = coerce_smr_agent_harness(
            mapping.get("agent_harness"),
            field_name="roles.binding.agent_harness",
        )
        kind = coerce_smr_agent_harness(
            mapping.get("agent_kind"),
            field_name="roles.binding.agent_kind",
        )
        if harness is not None and kind is not None and harness != kind:
            raise ValueError(
                "roles.binding.agent_harness and roles.binding.agent_kind must match when both are provided"
            )
        return cls(
            model=model,
            params=params or {},
            agent_harness=harness or kind,
            provider=RoleProviderRequirement.from_wire(
                mapping.get("provider"),
                field_name="roles.binding.provider",
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model.value}
        if self.params:
            payload["params"] = dict(self.params)
        if self.agent_harness is not None:
            payload["agent_harness"] = self.agent_harness.value
        if self.provider is not None:
            payload["provider"] = self.provider.to_wire()
        return payload


@dataclass(frozen=True, slots=True)
class WorkerRolePalette:
    permitted_models: tuple[SmrAgentModel, ...]
    default_model: SmrAgentModel
    default_params: dict[str, Any] = field(default_factory=dict)
    subtypes: dict[SmrWorkerSubtype, RoleBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.permitted_models:
            raise ValueError("roles.worker.permitted_models is required")
        seen_models: set[str] = set()
        for model in self.permitted_models:
            if model.value in seen_models:
                raise ValueError("roles.worker.permitted_models must not contain duplicates")
            seen_models.add(model.value)
        if self.default_model.value not in seen_models:
            raise ValueError(
                "roles.worker.default_model must be included in roles.worker.permitted_models"
            )
        normalized_default_params = _optional_mapping(
            self.default_params,
            field_name="roles.worker.default_params",
        )
        object.__setattr__(self, "default_params", normalized_default_params or {})
        normalized_subtypes: dict[SmrWorkerSubtype, RoleBinding] = {}
        for raw_subtype, raw_binding in dict(self.subtypes or {}).items():
            subtype = _coerce_worker_subtype(
                raw_subtype,
                field_name="roles.worker.subtypes",
            )
            binding = (
                raw_binding
                if isinstance(raw_binding, RoleBinding)
                else RoleBinding.from_wire(raw_binding)
            )
            if binding.model.value not in seen_models:
                raise ValueError(
                    f"roles.worker.subtypes.{subtype.value}.model must be included in roles.worker.permitted_models"
                )
            normalized_subtypes[subtype] = binding
        object.__setattr__(self, "subtypes", normalized_subtypes)

    @classmethod
    def from_wire(cls, payload: object) -> WorkerRolePalette:
        mapping = _require_mapping(payload, field_name="roles.worker")
        raw_permitted = mapping.get("permitted_models")
        if not isinstance(raw_permitted, list) or not raw_permitted:
            raise ValueError("roles.worker.permitted_models is required")
        permitted_models: list[SmrAgentModel] = []
        for index, item in enumerate(raw_permitted):
            model = coerce_smr_agent_model(
                item,
                field_name=f"roles.worker.permitted_models[{index}]",
            )
            if model is None:
                raise ValueError(f"roles.worker.permitted_models[{index}] must be a model value")
            permitted_models.append(model)
        default_model = coerce_smr_agent_model(
            mapping.get("default_model"),
            field_name="roles.worker.default_model",
        )
        if default_model is None:
            raise ValueError("roles.worker.default_model is required")
        default_params = _optional_mapping(
            mapping.get("default_params"),
            field_name="roles.worker.default_params",
        )
        raw_subtypes = mapping.get("subtypes")
        if raw_subtypes is None:
            raw_subtypes = {}
        if not isinstance(raw_subtypes, Mapping):
            raise ValueError("roles.worker.subtypes must be a mapping when provided")
        subtypes: dict[SmrWorkerSubtype, RoleBinding] = {}
        for raw_subtype, raw_binding in raw_subtypes.items():
            subtype = _coerce_worker_subtype(
                str(raw_subtype),
                field_name="roles.worker.subtypes",
            )
            subtypes[subtype] = RoleBinding.from_wire(raw_binding)
        return cls(
            permitted_models=tuple(permitted_models),
            default_model=default_model,
            default_params=default_params or {},
            subtypes=subtypes,
        )

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "permitted_models": [model.value for model in self.permitted_models],
            "default_model": self.default_model.value,
        }
        if self.default_params:
            payload["default_params"] = dict(self.default_params)
        if self.subtypes:
            payload["subtypes"] = {
                subtype.value: binding.to_wire() for subtype, binding in self.subtypes.items()
            }
        return payload


@dataclass(frozen=True, slots=True)
class SmrRoleBindings:
    orchestrator: RoleBinding
    reviewer: RoleBinding
    worker: WorkerRolePalette

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "orchestrator",
            self.orchestrator
            if isinstance(self.orchestrator, RoleBinding)
            else RoleBinding.from_wire(self.orchestrator),
        )
        object.__setattr__(
            self,
            "reviewer",
            self.reviewer
            if isinstance(self.reviewer, RoleBinding)
            else RoleBinding.from_wire(self.reviewer),
        )
        object.__setattr__(
            self,
            "worker",
            self.worker
            if isinstance(self.worker, WorkerRolePalette)
            else WorkerRolePalette.from_wire(self.worker),
        )

    @classmethod
    def from_wire(cls, payload: object) -> SmrRoleBindings:
        mapping = _require_mapping(payload, field_name="roles")
        return cls(
            orchestrator=RoleBinding.from_wire(mapping.get("orchestrator")),
            reviewer=RoleBinding.from_wire(mapping.get("reviewer")),
            worker=WorkerRolePalette.from_wire(mapping.get("worker")),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "orchestrator": self.orchestrator.to_wire(),
            "reviewer": self.reviewer.to_wire(),
            "worker": self.worker.to_wire(),
        }


def coerce_smr_role_bindings(
    value: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None,
    *,
    field_name: str = "roles",
) -> SmrRoleBindings | None:
    if value is None:
        return None
    if isinstance(value, SmrRoleBindings):
        return value
    if isinstance(value, Mapping):
        return SmrRoleBindings.from_wire(dict(value))
    raise ValueError(f"{field_name} must be a SmrRoleBindings or mapping when provided")


__all__ = [
    "RoleBinding",
    "RoleProviderRequirement",
    "SmrRoleBindings",
    "WorkerRolePalette",
    "coerce_smr_role_bindings",
]
