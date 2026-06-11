"""Launch payload helpers for the Managed Research SDK."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.smr_agent_harnesses import (
    SmrAgentHarness,
    coerce_smr_agent_harness,
)
from synth_ai.managed_research.models.smr_agent_kinds import SmrAgentKind, coerce_smr_agent_kind
from synth_ai.managed_research.models.smr_agent_models import SmrAgentModel
from synth_ai.managed_research.models.smr_funding_sources import (
    SmrFundingSource,
    coerce_smr_funding_source,
)
from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind, coerce_smr_host_kind
from synth_ai.managed_research.models.smr_horizons import (
    SmrIntendedHorizonHours,
    coerce_intended_horizon_hours,
)
from synth_ai.managed_research.models.smr_providers import (
    ProviderBinding,
    UsageLimit,
    coerce_provider_bindings,
    coerce_usage_limit,
)
from synth_ai.managed_research.models.smr_roles import (
    SmrRoleBindings,
    coerce_smr_role_bindings,
)
from synth_ai.managed_research.models.smr_run_policy import SmrRunPolicy, coerce_smr_run_policy
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode, coerce_smr_work_mode


def _optional_mapping(
    payload: Mapping[str, Any] | dict[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return dict(payload)


def _normalize_uploaded_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = str(entry.get("path") or "").strip()
    uri = str(entry.get("uri") or "").strip()
    name = str(entry.get("name") or "").strip()
    if not path and not uri:
        raise ValueError("uploaded file entries require path or uri")
    payload: dict[str, Any] = {}
    if path:
        payload["path"] = path
    if uri:
        payload["uri"] = uri
    if name:
        payload["name"] = name
    content_type = str(entry.get("content_type") or "").strip()
    if content_type:
        payload["content_type"] = content_type
    description = str(entry.get("description") or "").strip()
    if description:
        payload["description"] = description
    return payload


def normalize_resource_uploaded_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_uploaded_file(entry)
    visibility = str(entry.get("visibility") or "").strip()
    if visibility:
        normalized["visibility"] = visibility
    metadata = entry.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("resource file metadata must be a mapping when provided")
        normalized["metadata"] = dict(metadata)
    return normalized


def build_project_run_payload(
    *,
    objective: str | None = None,
    prompt: str | None = None,
    title: str | None = None,
    project_id: str | None = None,
    project_alias: str | None = None,
    task_id: str | None = None,
    task_key: str | None = None,
    run_idempotency_key: str | None = None,
    branch_name: str | None = None,
    runtime_label: str | None = None,
    notes: str | None = None,
    execution: Mapping[str, Any] | dict[str, Any] | None = None,
    metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    host_kind: SmrHostKind | str | None = None,
    work_mode: SmrWorkMode | str | None = None,
    mode: SmrWorkMode | str | None = None,
    intended_horizon_hours: SmrIntendedHorizonHours | int | None = None,
    provider_bindings: ProviderBinding
    | Mapping[str, Any]
    | list[ProviderBinding | Mapping[str, Any]]
    | None = None,
    funding_source: SmrFundingSource | str | None = None,
    usage_limit: UsageLimit | Mapping[str, Any] | None = None,
    run_policy: SmrRunPolicy | str | None = None,
    agent_profile: str | None = None,
    agent_model: SmrAgentModel | str | None = None,
    agent_model_params: Mapping[str, Any] | dict[str, Any] | None = None,
    agent_harness: SmrAgentHarness | str | None = None,
    agent_kind: SmrAgentKind | str | None = None,
    actor_model_overrides: Mapping[str, Any]
    | dict[str, Any]
    | list[Mapping[str, Any]]
    | None = None,
    roles: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None = None,
    uploaded_files: list[Mapping[str, Any]] | None = None,
    resource_files: list[Mapping[str, Any]] | None = None,
    one_off: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in (
        ("objective", objective),
        ("prompt", prompt),
        ("title", title),
        ("project_id", project_id),
        ("project_alias", project_alias),
        ("task_id", task_id),
        ("task_key", task_key),
        ("run_idempotency_key", run_idempotency_key),
        ("branch_name", branch_name),
        ("runtime_label", runtime_label),
        ("notes", notes),
    ):
        text = str(value or "").strip()
        if text:
            payload[key] = text

    normalized_execution = _optional_mapping(execution, field_name="execution")
    if normalized_execution is not None:
        payload["execution"] = normalized_execution
    normalized_metadata = _optional_mapping(metadata, field_name="metadata")
    if normalized_metadata is not None:
        payload["metadata"] = normalized_metadata

    normalized_host_kind = coerce_smr_host_kind(host_kind, field_name="host_kind")
    if normalized_host_kind is not None:
        payload["host_kind"] = normalized_host_kind.value
    if work_mode is not None and mode is not None:
        normalized_work_mode = coerce_smr_work_mode(work_mode, field_name="work_mode")
        normalized_mode = coerce_smr_work_mode(mode, field_name="mode")
        if normalized_work_mode != normalized_mode:
            raise ValueError("work_mode and mode must match when both are provided")
    else:
        normalized_work_mode = coerce_smr_work_mode(
            work_mode if work_mode is not None else mode,
            field_name="work_mode",
        )
    if normalized_work_mode is not None:
        payload["work_mode"] = normalized_work_mode.value
    normalized_horizon = coerce_intended_horizon_hours(
        intended_horizon_hours,
        field_name="intended_horizon_hours",
    )
    if normalized_horizon is not None:
        payload["intended_horizon_hours"] = int(normalized_horizon)
    if provider_bindings is not None:
        normalized_provider_bindings = coerce_provider_bindings(
            provider_bindings,
            field_name="provider_bindings",
        )
        if normalized_provider_bindings:
            payload["provider_bindings"] = [
                binding.as_payload() for binding in normalized_provider_bindings
            ]
    normalized_funding_source = coerce_smr_funding_source(
        funding_source,
        field_name="funding_source",
    )
    if normalized_funding_source is not None:
        payload["funding_source"] = normalized_funding_source.value
    normalized_usage_limit = coerce_usage_limit(usage_limit, field_name="usage_limit")
    if normalized_usage_limit is not None:
        payload["usage_limit"] = normalized_usage_limit.as_payload()
    normalized_run_policy = coerce_smr_run_policy(run_policy, field_name="run_policy")
    if normalized_run_policy is not None:
        payload["run_policy"] = normalized_run_policy.value
    if str(agent_profile or "").strip():
        payload["agent_profile"] = str(agent_profile).strip()
    if str(agent_model or "").strip():
        payload["agent_model"] = str(agent_model).strip()
    normalized_agent_model_params = _optional_mapping(
        agent_model_params,
        field_name="agent_model_params",
    )
    if normalized_agent_model_params is not None:
        payload["agent_model_params"] = normalized_agent_model_params
    normalized_agent_kind = coerce_smr_agent_kind(agent_kind, field_name="agent_kind")
    normalized_agent_harness = coerce_smr_agent_harness(
        agent_harness,
        field_name="agent_harness",
    )
    if (
        normalized_agent_harness is not None
        and normalized_agent_kind is not None
        and normalized_agent_harness.value != normalized_agent_kind.value
    ):
        raise ValueError("agent_harness and agent_kind must match when both are provided")
    resolved_agent_harness = normalized_agent_harness or normalized_agent_kind
    if resolved_agent_harness is not None:
        payload["agent_harness"] = resolved_agent_harness.value
    normalized_roles = coerce_smr_role_bindings(roles, field_name="roles")
    if normalized_roles is not None:
        if actor_model_overrides:
            raise ValueError("roles cannot be combined with actor_model_overrides")
        if any(
            (
                str(agent_profile or "").strip(),
                str(agent_model or "").strip(),
                normalized_agent_model_params is not None,
                resolved_agent_harness is not None,
            )
        ):
            raise ValueError(
                "roles cannot be combined with shared top-level "
                "agent_profile/agent_model/agent_harness/agent_kind/agent_model_params"
            )
        payload["roles"] = normalized_roles.to_wire()
    if actor_model_overrides is not None:
        payload["actor_model_overrides"] = actor_model_overrides
    if uploaded_files is not None:
        payload["uploaded_files"] = [_normalize_uploaded_file(item) for item in uploaded_files]
    if resource_files is not None:
        payload["resource_files"] = [
            normalize_resource_uploaded_file(item) for item in resource_files
        ]
    if one_off:
        payload["one_off"] = True
    return payload


__all__ = [
    "build_project_run_payload",
    "normalize_resource_uploaded_file",
]
