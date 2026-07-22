"""Typed request parsing helpers for MCP tool handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synth_ai.mcp.research.registry import JSONDict
from synth_ai.core.research._legacy.models.run_timeline import SmrBranchMode, SmrRunBranchRequest
from synth_ai.core.research._legacy.models.smr_evidence_obligations import (
    EvidenceObligations,
    coerce_evidence_obligations,
)
from synth_ai.core.research._legacy.models.smr_actor_models import normalize_actor_model_assignments
from synth_ai.core.research._legacy.models.smr_agent_harnesses import coerce_smr_agent_harness
from synth_ai.core.research._legacy.models.smr_agent_kinds import coerce_smr_agent_kind
from synth_ai.core.research._legacy.models.smr_agent_models import coerce_smr_agent_model
from synth_ai.core.research._legacy.models.smr_credential_providers import (
    coerce_smr_credential_provider,
)
from synth_ai.core.research._legacy.models.smr_funding_sources import coerce_smr_funding_source
from synth_ai.core.research._legacy.models.smr_horizons import coerce_intended_horizon_hours
from synth_ai.core.research._legacy.models.smr_host_kinds import coerce_smr_host_kind
from synth_ai.core.research._legacy.models.smr_providers import (
    coerce_provider_bindings,
    coerce_usage_limit,
)
from synth_ai.core.research._legacy.models.smr_roles import coerce_smr_role_bindings
from synth_ai.core.research._legacy.models.smr_run_policy import coerce_smr_run_policy
from synth_ai.core.research._legacy.models.smr_runbooks import coerce_smr_runbook_kind
from synth_ai.core.research._legacy.models.smr_work_modes import coerce_smr_work_mode
from synth_ai.core.research._legacy.models.types import SmrRunnableProjectRequest


def require_string(payload: JSONDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return value.strip()


def optional_string(payload: JSONDict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string when provided")
    stripped = value.strip()
    return stripped or None


def optional_int(payload: JSONDict, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer when provided")
    return value


def optional_intended_horizon_hours(payload: JSONDict) -> int | None:
    value = optional_int(payload, "intended_horizon_hours")
    horizon = coerce_intended_horizon_hours(value, field_name="intended_horizon_hours")
    return int(horizon) if horizon is not None else None


def optional_bool(payload: JSONDict, key: str, *, default: bool = False) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean when provided")
    return value


def parse_branch_run_request(payload: JSONDict) -> SmrRunBranchRequest:
    raw_mode = optional_string(payload, "mode")
    mode = SmrBranchMode(raw_mode) if raw_mode is not None else SmrBranchMode.EXACT
    return SmrRunBranchRequest(
        checkpoint_id=optional_string(payload, "checkpoint_id"),
        checkpoint_record_id=optional_string(payload, "checkpoint_record_id"),
        checkpoint_uri=optional_string(payload, "checkpoint_uri"),
        mode=mode,
        message=optional_string(payload, "message"),
        reason=optional_string(payload, "reason"),
        title=optional_string(payload, "title"),
        source_node_id=optional_string(payload, "source_node_id"),
    )


def require_smr_work_mode(payload: JSONDict, key: str) -> str:
    value = require_string(payload, key)
    work_mode = coerce_smr_work_mode(value, field_name=key)
    if work_mode is None:
        raise ValueError(f"'{key}' is required")
    return work_mode.value


def optional_smr_work_mode(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    work_mode = coerce_smr_work_mode(value, field_name=key)
    return work_mode.value if work_mode is not None else None


def require_smr_host_kind(payload: JSONDict, key: str) -> str:
    value = require_string(payload, key)
    host_kind = coerce_smr_host_kind(value, field_name=key)
    if host_kind is None:
        raise ValueError(f"'{key}' is required")
    return host_kind.value


def optional_smr_host_kind(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    host_kind = coerce_smr_host_kind(value, field_name=key)
    return host_kind.value if host_kind is not None else None


def optional_smr_runbook_kind(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    runbook = coerce_smr_runbook_kind(value, field_name=key)
    return runbook.value if runbook is not None else None


def require_smr_credential_provider(payload: JSONDict, key: str) -> str:
    value = require_string(payload, key)
    provider = coerce_smr_credential_provider(value, field_name=key)
    if provider is None:
        raise ValueError(f"'{key}' is required")
    return provider.value


def require_smr_funding_source(payload: JSONDict, key: str) -> str:
    value = require_string(payload, key)
    funding_source = coerce_smr_funding_source(value, field_name=key)
    if funding_source is None:
        raise ValueError(f"'{key}' is required")
    return funding_source.value


def optional_smr_agent_model(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    model = coerce_smr_agent_model(value, field_name=key)
    return model.value if model is not None else None


def optional_smr_agent_kind(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    agent_kind = coerce_smr_agent_kind(value, field_name=key)
    return agent_kind.value if agent_kind is not None else None


def optional_smr_agent_harness(payload: JSONDict, key: str) -> str | None:
    value = optional_string(payload, key)
    agent_harness = coerce_smr_agent_harness(value, field_name=key)
    return agent_harness.value if agent_harness is not None else None


def optional_smr_run_policy(payload: JSONDict, key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object when provided")
    return coerce_smr_run_policy(value, field_name=key)


def require_provider_bindings(payload: JSONDict, key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    return [binding.to_wire() for binding in coerce_provider_bindings(value, field_name=key)]


def optional_provider_bindings(payload: JSONDict, key: str) -> list[dict[str, Any]] | None:
    value = payload.get(key)
    if value is None:
        return None
    return [binding.to_wire() for binding in coerce_provider_bindings(value, field_name=key)]


def optional_usage_limit(payload: JSONDict, key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    limit = coerce_usage_limit(value, field_name=key)
    if limit is None:
        return None
    payload = limit.to_dict()
    return payload or None


def optional_actor_model_assignments(payload: JSONDict, key: str) -> list[dict[str, Any]] | None:
    value = payload.get(key)
    normalized = normalize_actor_model_assignments(value, field_name=key)
    if not normalized:
        return None
    return [item.as_payload() for item in normalized]


def optional_role_bindings(payload: JSONDict, key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    normalized = coerce_smr_role_bindings(value, field_name=key)
    if normalized is None:
        return None
    return normalized.to_wire()


def optional_evidence_obligations(
    payload: JSONDict,
    key: str = "evidence_obligations",
) -> EvidenceObligations | None:
    return coerce_evidence_obligations(payload.get(key), field_name=key)


def reject_legacy_prompt_arg(payload: JSONDict) -> None:
    if "prompt" in payload:
        raise ValueError(
            "The `prompt` field is no longer supported; use "
            "`initial_runtime_messages` to enqueue kickoff text on the runtime "
            "message queue."
        )


def _optional_object(payload: JSONDict, key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object when provided")
    return dict(value)


def _optional_object_list(payload: JSONDict, key: str) -> list[dict[str, Any]] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be an array when provided")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"'{key}' entries must be objects")
        normalized.append(dict(item))
    return normalized or None


def _validate_runbook_preset_aliases(
    *,
    runbook_preset: str | None,
    runbook_config_id: str | None,
) -> None:
    if runbook_preset and runbook_config_id:
        raise ValueError("runbook_preset and runbook_config_id cannot both be provided")


def _requires_explicit_launch_axes(
    *,
    runbook_preset: str | None,
    runbook_config_id: str | None,
    intended_horizon_hours: int | None,
) -> bool:
    return not (runbook_preset or runbook_config_id or intended_horizon_hours is not None)


def _require_explicit_launch_axes(
    *,
    host_kind: str | None,
    work_mode: str | None,
    providers: list[dict[str, Any]] | None,
) -> None:
    missing: list[str] = []
    if host_kind is None:
        missing.append("host_kind")
    if work_mode is None:
        missing.append("work_mode")
    if providers is None:
        missing.append("providers")
    if missing:
        raise ValueError(
            "Provide intended_horizon_hours, runbook_preset, or explicit launch fields: "
            + ", ".join(missing)
        )


@dataclass(frozen=True)
class ProjectMutationRequest:
    project_id: str
    config: dict[str, Any]
    actor_model_assignments: list[dict[str, Any]] | None = None

    @classmethod
    def for_create(cls, payload: JSONDict) -> ProjectMutationRequest:
        raw_config = payload.get("config")
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, dict):
            config = dict(raw_config)
        else:
            raise ValueError("'config' must be an object when provided")
        name = optional_string(payload, "name")
        if name is not None:
            config["name"] = name
        return cls(
            project_id="",
            config=config,
            actor_model_assignments=optional_actor_model_assignments(
                payload, "actor_model_assignments"
            ),
        )

    @classmethod
    def for_patch(cls, payload: JSONDict) -> ProjectMutationRequest:
        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("'config' is required and must be an object")
        return cls(
            project_id=require_string(payload, "project_id"),
            config=dict(raw_config),
            actor_model_assignments=optional_actor_model_assignments(
                payload, "actor_model_assignments"
            ),
        )


@dataclass(frozen=True)
class RunnableProjectCreateRequest:
    request: SmrRunnableProjectRequest

    @classmethod
    def from_payload(cls, payload: JSONDict) -> RunnableProjectCreateRequest:
        normalized = dict(payload)
        agent_profiles: dict[str, Any] = {
            "orchestrator_profile_id": require_string(payload, "orchestrator_profile_id"),
            "default_worker_profile_id": require_string(payload, "default_worker_profile_id"),
        }
        worker_profile_ids = payload.get("worker_profile_ids")
        if worker_profile_ids is not None:
            if not isinstance(worker_profile_ids, list):
                raise ValueError("'worker_profile_ids' must be an array when provided")
            agent_profiles["worker_profile_ids"] = [
                require_string({"value": item}, "value") for item in worker_profile_ids
            ]
        normalized["agent_profiles"] = agent_profiles
        return cls(request=SmrRunnableProjectRequest.from_wire(normalized))


@dataclass(frozen=True)
class ProviderKeyRequest:
    project_id: str
    provider: str
    funding_source: str
    api_key: str | None = None
    encrypted_key_b64: str | None = None

    @classmethod
    def from_payload(cls, payload: JSONDict) -> ProviderKeyRequest:
        return cls(
            project_id=require_string(payload, "project_id"),
            provider=require_smr_credential_provider(payload, "provider"),
            funding_source=require_smr_funding_source(payload, "funding_source"),
            api_key=optional_string(payload, "api_key"),
            encrypted_key_b64=optional_string(payload, "encrypted_key_b64"),
        )


@dataclass(frozen=True)
class RunLaunchRequest:
    project_id: str | None
    host_kind: str | None
    work_mode: str | None
    intended_horizon_hours: int | None
    providers: list[dict[str, Any]] | None
    objective: str | None = None
    runbook: str | None = None
    runbook_preset: str | None = None
    runbook_config_id: str | None = None
    limit: dict[str, Any] | None = None
    worker_pool_id: str | None = None
    timebox_seconds: int | None = None
    agent_profile: str | None = None
    agent_model: str | None = None
    agent_harness: str | None = None
    agent_kind: str | None = None
    agent_model_params: dict[str, Any] | None = None
    actor_model_overrides: list[dict[str, Any]] | None = None
    roles: dict[str, Any] | None = None
    initial_runtime_messages: list[dict[str, Any]] | None = None
    workflow: dict[str, Any] | None = None
    sandbox_override: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    dev_environment_id: str | None = None
    local_execution: dict[str, Any] | None = None
    execution_profile: dict[str, Any] | None = None
    run_policy: dict[str, Any] | None = None
    kickoff_contract: dict[str, Any] | None = None
    resource_bindings: dict[str, Any] | None = None
    evidence_obligations: EvidenceObligations | None = None
    open_ended_question: dict[str, Any] | None = None
    directed_effort_outcome: dict[str, Any] | None = None
    required_work_products: list[dict[str, Any]] | None = None
    require_report: bool = True
    ai_cache: dict[str, Any] | None = None
    primary_objective_id: str | None = None
    primary_objective_kind: str | None = None
    primary_parent_ref: dict[str, Any] | None = None
    primary_parent: dict[str, Any] | None = None
    effort_id: str | None = None
    idempotency_key_run_create: str | None = None
    idempotency_key: str | None = None

    @classmethod
    def from_payload(cls, payload: JSONDict) -> RunLaunchRequest:
        reject_legacy_prompt_arg(payload)
        runbook_preset = optional_string(payload, "runbook_preset")
        runbook_config_id = optional_string(payload, "runbook_config_id")
        _validate_runbook_preset_aliases(
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
        )
        host_kind = optional_smr_host_kind(payload, "host_kind")
        work_mode = optional_smr_work_mode(payload, "work_mode")
        mode = optional_smr_work_mode(payload, "mode")
        if work_mode is not None and mode is not None and work_mode != mode:
            raise ValueError("work_mode and mode must match when both are provided")
        work_mode = work_mode or mode
        intended_horizon_hours = optional_intended_horizon_hours(payload)
        providers = optional_provider_bindings(payload, "providers")
        if _requires_explicit_launch_axes(
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            intended_horizon_hours=intended_horizon_hours,
        ):
            _require_explicit_launch_axes(
                host_kind=host_kind,
                work_mode=work_mode,
                providers=providers,
            )
        return cls(
            project_id=optional_string(payload, "project_id"),
            host_kind=host_kind,
            work_mode=work_mode,
            intended_horizon_hours=intended_horizon_hours,
            providers=providers,
            objective=optional_string(payload, "objective"),
            runbook=optional_smr_runbook_kind(payload, "runbook"),
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            limit=optional_usage_limit(payload, "limit"),
            worker_pool_id=optional_string(payload, "worker_pool_id"),
            timebox_seconds=optional_int(payload, "timebox_seconds"),
            agent_profile=optional_string(payload, "agent_profile"),
            agent_model=optional_smr_agent_model(payload, "agent_model"),
            agent_harness=optional_smr_agent_harness(payload, "agent_harness"),
            agent_kind=optional_smr_agent_kind(payload, "agent_kind"),
            agent_model_params=_optional_object(payload, "agent_model_params"),
            actor_model_overrides=optional_actor_model_assignments(
                payload, "actor_model_overrides"
            ),
            roles=optional_role_bindings(payload, "roles"),
            initial_runtime_messages=_optional_object_list(payload, "initial_runtime_messages"),
            workflow=_optional_object(payload, "workflow"),
            sandbox_override=_optional_object(payload, "sandbox_override"),
            environment=_optional_object(payload, "environment"),
            dev_environment_id=optional_string(payload, "dev_environment_id"),
            local_execution=_optional_object(payload, "local_execution"),
            execution_profile=_optional_object(payload, "execution_profile"),
            run_policy=optional_smr_run_policy(payload, "run_policy"),
            kickoff_contract=_optional_object(payload, "kickoff_contract"),
            resource_bindings=_optional_object(payload, "resource_bindings"),
            evidence_obligations=optional_evidence_obligations(payload),
            open_ended_question=_optional_object(payload, "open_ended_question"),
            directed_effort_outcome=_optional_object(payload, "directed_effort_outcome"),
            required_work_products=_optional_object_list(payload, "required_work_products"),
            require_report=optional_bool(payload, "require_report", default=True),
            ai_cache=_optional_object(payload, "ai_cache"),
            primary_objective_id=optional_string(payload, "primary_objective_id"),
            primary_objective_kind=optional_string(payload, "primary_objective_kind"),
            primary_parent_ref=_optional_object(payload, "primary_parent_ref"),
            primary_parent=_optional_object(payload, "primary_parent"),
            effort_id=optional_string(payload, "effort_id"),
            idempotency_key_run_create=optional_string(payload, "idempotency_key_run_create"),
            idempotency_key=optional_string(payload, "idempotency_key"),
        )

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "host_kind": self.host_kind,
            "work_mode": self.work_mode,
            "intended_horizon_hours": self.intended_horizon_hours,
            "providers": self.providers,
            "objective": self.objective,
            "runbook": self.runbook,
            "runbook_preset": self.runbook_preset,
            "runbook_config_id": self.runbook_config_id,
            "limit": self.limit,
            "worker_pool_id": self.worker_pool_id,
            "timebox_seconds": self.timebox_seconds,
            "agent_profile": self.agent_profile,
            "agent_model": self.agent_model,
            "agent_harness": self.agent_harness,
            "agent_kind": self.agent_kind,
            "agent_model_params": self.agent_model_params,
            "actor_model_overrides": self.actor_model_overrides,
            "roles": self.roles,
            "initial_runtime_messages": self.initial_runtime_messages,
            "workflow": self.workflow,
            "sandbox_override": self.sandbox_override,
            "environment": self.environment,
            "dev_environment_id": self.dev_environment_id,
            "local_execution": self.local_execution,
            "execution_profile": self.execution_profile,
            "run_policy": self.run_policy,
            "kickoff_contract": self.kickoff_contract,
            "resource_bindings": self.resource_bindings,
            "evidence_obligations": self.evidence_obligations,
            "open_ended_question": self.open_ended_question,
            "directed_effort_outcome": self.directed_effort_outcome,
            "required_work_products": self.required_work_products,
            "require_report": self.require_report,
            "ai_cache": self.ai_cache,
            "primary_objective_id": self.primary_objective_id,
            "primary_objective_kind": self.primary_objective_kind,
            "primary_parent_ref": self.primary_parent_ref,
            "primary_parent": self.primary_parent,
            "effort_id": self.effort_id,
            "idempotency_key_run_create": self.idempotency_key_run_create,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class OneOffRunLaunchRequest:
    host_kind: str | None
    work_mode: str | None
    intended_horizon_hours: int | None
    providers: list[dict[str, Any]] | None
    objective: str | None = None
    runbook: str | None = None
    runbook_preset: str | None = None
    runbook_config_id: str | None = None
    limit: dict[str, Any] | None = None
    worker_pool_id: str | None = None
    timebox_seconds: int | None = None
    agent_profile: str | None = None
    agent_model: str | None = None
    agent_harness: str | None = None
    agent_kind: str | None = None
    agent_model_params: dict[str, Any] | None = None
    actor_model_overrides: list[dict[str, Any]] | None = None
    roles: dict[str, Any] | None = None
    initial_runtime_messages: list[dict[str, Any]] | None = None
    workflow: dict[str, Any] | None = None
    sandbox_override: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    dev_environment_id: str | None = None
    local_execution: dict[str, Any] | None = None
    execution_profile: dict[str, Any] | None = None
    run_policy: dict[str, Any] | None = None
    kickoff_contract: dict[str, Any] | None = None
    resource_bindings: dict[str, Any] | None = None
    evidence_obligations: EvidenceObligations | None = None
    open_ended_question: dict[str, Any] | None = None
    directed_effort_outcome: dict[str, Any] | None = None
    required_work_products: list[dict[str, Any]] | None = None
    require_report: bool = True
    ai_cache: dict[str, Any] | None = None
    primary_objective_id: str | None = None
    primary_objective_kind: str | None = None
    primary_parent_ref: dict[str, Any] | None = None
    primary_parent: dict[str, Any] | None = None
    effort_id: str | None = None
    idempotency_key_run_create: str | None = None
    idempotency_key: str | None = None

    @classmethod
    def from_payload(cls, payload: JSONDict) -> OneOffRunLaunchRequest:
        reject_legacy_prompt_arg(payload)
        runbook_preset = optional_string(payload, "runbook_preset")
        runbook_config_id = optional_string(payload, "runbook_config_id")
        _validate_runbook_preset_aliases(
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
        )
        host_kind = optional_smr_host_kind(payload, "host_kind")
        work_mode = optional_smr_work_mode(payload, "work_mode")
        mode = optional_smr_work_mode(payload, "mode")
        if work_mode is not None and mode is not None and work_mode != mode:
            raise ValueError("work_mode and mode must match when both are provided")
        work_mode = work_mode or mode
        intended_horizon_hours = optional_intended_horizon_hours(payload)
        providers = optional_provider_bindings(payload, "providers")
        if _requires_explicit_launch_axes(
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            intended_horizon_hours=intended_horizon_hours,
        ):
            _require_explicit_launch_axes(
                host_kind=host_kind,
                work_mode=work_mode,
                providers=providers,
            )
        return cls(
            host_kind=host_kind,
            work_mode=work_mode,
            intended_horizon_hours=intended_horizon_hours,
            providers=providers,
            objective=optional_string(payload, "objective"),
            runbook=optional_smr_runbook_kind(payload, "runbook"),
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            limit=optional_usage_limit(payload, "limit"),
            worker_pool_id=optional_string(payload, "worker_pool_id"),
            timebox_seconds=optional_int(payload, "timebox_seconds"),
            agent_profile=optional_string(payload, "agent_profile"),
            agent_model=optional_smr_agent_model(payload, "agent_model"),
            agent_harness=optional_smr_agent_harness(payload, "agent_harness"),
            agent_kind=optional_smr_agent_kind(payload, "agent_kind"),
            agent_model_params=_optional_object(payload, "agent_model_params"),
            actor_model_overrides=optional_actor_model_assignments(
                payload, "actor_model_overrides"
            ),
            roles=optional_role_bindings(payload, "roles"),
            initial_runtime_messages=_optional_object_list(payload, "initial_runtime_messages"),
            workflow=_optional_object(payload, "workflow"),
            sandbox_override=_optional_object(payload, "sandbox_override"),
            environment=_optional_object(payload, "environment"),
            dev_environment_id=optional_string(payload, "dev_environment_id"),
            local_execution=_optional_object(payload, "local_execution"),
            execution_profile=_optional_object(payload, "execution_profile"),
            run_policy=optional_smr_run_policy(payload, "run_policy"),
            kickoff_contract=_optional_object(payload, "kickoff_contract"),
            resource_bindings=_optional_object(payload, "resource_bindings"),
            evidence_obligations=optional_evidence_obligations(payload),
            open_ended_question=_optional_object(payload, "open_ended_question"),
            directed_effort_outcome=_optional_object(payload, "directed_effort_outcome"),
            required_work_products=_optional_object_list(payload, "required_work_products"),
            require_report=optional_bool(payload, "require_report", default=True),
            ai_cache=_optional_object(payload, "ai_cache"),
            primary_objective_id=optional_string(payload, "primary_objective_id"),
            primary_objective_kind=optional_string(payload, "primary_objective_kind"),
            primary_parent_ref=_optional_object(payload, "primary_parent_ref"),
            primary_parent=_optional_object(payload, "primary_parent"),
            effort_id=optional_string(payload, "effort_id"),
            idempotency_key_run_create=optional_string(payload, "idempotency_key_run_create"),
            idempotency_key=optional_string(payload, "idempotency_key"),
        )

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "host_kind": self.host_kind,
            "work_mode": self.work_mode,
            "intended_horizon_hours": self.intended_horizon_hours,
            "providers": self.providers,
            "objective": self.objective,
            "runbook": self.runbook,
            "runbook_preset": self.runbook_preset,
            "runbook_config_id": self.runbook_config_id,
            "limit": self.limit,
            "worker_pool_id": self.worker_pool_id,
            "timebox_seconds": self.timebox_seconds,
            "agent_profile": self.agent_profile,
            "agent_model": self.agent_model,
            "agent_harness": self.agent_harness,
            "agent_kind": self.agent_kind,
            "agent_model_params": self.agent_model_params,
            "actor_model_overrides": self.actor_model_overrides,
            "roles": self.roles,
            "initial_runtime_messages": self.initial_runtime_messages,
            "workflow": self.workflow,
            "sandbox_override": self.sandbox_override,
            "environment": self.environment,
            "dev_environment_id": self.dev_environment_id,
            "local_execution": self.local_execution,
            "execution_profile": self.execution_profile,
            "run_policy": self.run_policy,
            "kickoff_contract": self.kickoff_contract,
            "resource_bindings": self.resource_bindings,
            "evidence_obligations": self.evidence_obligations,
            "open_ended_question": self.open_ended_question,
            "directed_effort_outcome": self.directed_effort_outcome,
            "required_work_products": self.required_work_products,
            "require_report": self.require_report,
            "ai_cache": self.ai_cache,
            "primary_objective_id": self.primary_objective_id,
            "primary_objective_kind": self.primary_objective_kind,
            "primary_parent_ref": self.primary_parent_ref,
            "primary_parent": self.primary_parent,
            "effort_id": self.effort_id,
            "idempotency_key_run_create": self.idempotency_key_run_create,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class UsageAnalyticsRequest:
    subject_kind: str | None
    org_id: str | None
    managed_account_id: str | None
    start_at: str
    end_at: str
    bucket: str
    first: int
    after: str | None

    @classmethod
    def from_payload(cls, payload: JSONDict) -> UsageAnalyticsRequest:
        first = optional_int(payload, "first")
        if first is None:
            raise ValueError("'first' is required")
        return cls(
            subject_kind=optional_string(payload, "subject_kind"),
            org_id=optional_string(payload, "org_id"),
            managed_account_id=optional_string(payload, "managed_account_id"),
            start_at=require_string(payload, "start_at"),
            end_at=require_string(payload, "end_at"),
            bucket=require_string(payload, "bucket").upper(),
            first=first,
            after=optional_string(payload, "after"),
        )
