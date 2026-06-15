"""Managed Research control-plane client."""

from __future__ import annotations

import base64
import json as _json
import mimetypes
import os
import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models import (
    BillingEntitlementSnapshot,
    Checkpoint,
    EffortCreateRequest,
    EffortPatchRequest,
    FactoryCreateRequest,
    FactoryPatchRequest,
    SmrProjectEconomics,
    SmrProjectUsage,
    SmrResourceLimitExtension,
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrResourceLimitSelector,
    SmrRunUsage,
)
from synth_ai.managed_research.models.factories import (
    effort_create_payload,
    effort_patch_payload,
    factory_create_payload,
    factory_patch_payload,
)
from synth_ai.managed_research.models.local_execution_profile import (
    LocalExecutionProfile,
    LocalPublicationReadiness,
)
from synth_ai.managed_research.models.run_control import ManagedResearchRunControlError
from synth_ai.managed_research.models.run_diagnostics import (
    SmrRunActorLogs,
    SmrRunActorUsage,
    SmrRunArtifactProgress,
    SmrRunCostSummary,
    SmrRunParticipants,
    SmrRunTraces,
)
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_execution import RunExecutionProjection
from synth_ai.managed_research.models.run_observability import (
    ManagedResearchRunContract,
    RunObservabilitySnapshot,
    RunObservationCursor,
)
from synth_ai.managed_research.models.run_state import ManagedResearchRun
from synth_ai.managed_research.models.run_timeline import (
    SmrAuthorityReadouts,
    SmrBranchMode,
    SmrLogicalTimeline,
    SmrRunBranchRequest,
    SmrRunBranchResponse,
    SmrRunEventLog,
)
from synth_ai.managed_research.models.runtime_intent import (
    RuntimeIntent,
    RuntimeIntentReceipt,
    RuntimeIntentView,
)
from synth_ai.managed_research.models.smr_actor_models import (
    SmrActorModelAssignment,
    normalize_actor_model_assignments,
    validate_shared_top_level_agent_model,
)
from synth_ai.managed_research.models.smr_agent_harnesses import (
    SmrAgentHarness,
    coerce_smr_agent_harness,
)
from synth_ai.managed_research.models.smr_agent_kinds import (
    SmrAgentKind,
    coerce_smr_agent_kind,
)
from synth_ai.managed_research.models.smr_agent_models import SmrAgentModel
from synth_ai.managed_research.models.smr_credential_providers import (
    SmrCredentialProvider,
    coerce_smr_credential_provider,
)
from synth_ai.managed_research.models.smr_funding_sources import (
    SmrFundingSource,
    coerce_smr_funding_source,
)
from synth_ai.managed_research.models.smr_horizons import (
    SmrIntendedHorizonHours,
    coerce_intended_horizon_hours,
)
from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind, coerce_smr_host_kind
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
from synth_ai.managed_research.models.smr_runbooks import (
    SmrRunbookKind,
    SmrRunbookPreset,
    coerce_smr_runbook_kind,
)
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode, coerce_smr_work_mode
from synth_ai.managed_research.models.types import (
    KickoffContract,
    RunArtifact,
    RunArtifactManifest,
    RunResourceBindings,
    SmrRunnableProjectRequest,
)
from synth_ai.managed_research.sdk.approvals import ApprovalsAPI
from synth_ai.managed_research.sdk.compat import SmrControlClientMixin
from synth_ai.managed_research.sdk.config import (
    DEFAULT_MISC_PROJECT_ALIAS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS,
    OPENAI_TRANSPORT_MODE_AUTO,
    OPENAI_TRANSPORT_MODE_BACKEND_BFF,
    OPENAI_TRANSPORT_MODE_DIRECT_HP,
)
from synth_ai.managed_research.sdk.config import (
    resolve_api_key as _resolve_api_key,
)
from synth_ai.managed_research.sdk.config import (
    resolve_backend_base as _resolve_backend_base,
)
from synth_ai.managed_research.sdk.cost import RunCostAPI
from synth_ai.managed_research.sdk.credentials import CredentialsAPI
from synth_ai.managed_research.sdk.datasets import DatasetsAPI
from synth_ai.managed_research.sdk.environments import EnvironmentsAPI
from synth_ai.managed_research.sdk.exports import ExportsAPI
from synth_ai.managed_research.sdk.factories import EffortsAPI, FactoriesAPI
from synth_ai.managed_research.sdk.files import FilesAPI
from synth_ai.managed_research.sdk.github import GithubAPI
from synth_ai.managed_research.sdk.integrations import IntegrationsAPI
from synth_ai.managed_research.sdk.logs import LogsAPI
from synth_ai.managed_research.sdk.models import ModelsAPI
from synth_ai.managed_research.sdk.outputs import OutputsAPI
from synth_ai.managed_research.sdk.progress import ProgressAPI
from synth_ai.managed_research.sdk.project import ManagedResearchProjectClient
from synth_ai.managed_research.sdk.projects import ProjectsAPI
from synth_ai.managed_research.sdk.prs import PrsAPI
from synth_ai.managed_research.sdk.readiness import ReadinessAPI
from synth_ai.managed_research.sdk.repos import ReposAPI
from synth_ai.managed_research.sdk.repositories import RepositoriesAPI
from synth_ai.managed_research.sdk.runs import RunsAPI
from synth_ai.managed_research.sdk.secrets import SecretsAPI
from synth_ai.managed_research.sdk.setup import SetupAPI
from synth_ai.managed_research.sdk.trained_models import TrainedModelsAPI
from synth_ai.managed_research.sdk.transport import build_http_transport
from synth_ai.managed_research.sdk.usage import UsageAPI
from synth_ai.managed_research.sdk.work_products import WorkProductsAPI
from synth_ai.managed_research.sdk.workspace_inputs import WorkspaceInputsAPI
from synth_ai.managed_research.transport.http import SmrHttpTransport, _raise_for_error_response
from synth_ai.managed_research.transport.pagination import build_query_params

ACTIVE_RUN_STATES = {"queued", "planning", "executing", "blocked", "finalizing", "running"}

__all__ = [
    "ACTIVE_RUN_STATES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MISC_PROJECT_ALIAS",
    "OPENAI_TRANSPORT_MODE_AUTO",
    "OPENAI_TRANSPORT_MODE_BACKEND_BFF",
    "OPENAI_TRANSPORT_MODE_DIRECT_HP",
    "ManagedResearchClient",
    "SmrControlClient",
    "first_id",
]


def _coerce_dict(payload: Any, *, label: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    raise SmrApiError(f"Expected object response for {label}, received {type(payload).__name__}")


def _coerce_dict_list(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise SmrApiError(f"Expected {label} entries to be objects")
        return cast(list[dict[str, Any]], payload)
    raise SmrApiError(f"Expected list response for {label}, received {type(payload).__name__}")


def _require_non_empty_string(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_non_empty_string(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


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


def _coerce_branch_request(
    *,
    checkpoint_id: str | None = None,
    checkpoint_record_id: str | None = None,
    checkpoint_uri: str | None = None,
    mode: SmrBranchMode | str = SmrBranchMode.EXACT,
    message: str | None = None,
    reason: str | None = None,
    title: str | None = None,
    source_node_id: str | None = None,
) -> SmrRunBranchRequest:
    normalized_mode = mode if isinstance(mode, SmrBranchMode) else SmrBranchMode(str(mode).strip())
    return SmrRunBranchRequest(
        checkpoint_id=checkpoint_id,
        checkpoint_record_id=checkpoint_record_id,
        checkpoint_uri=checkpoint_uri,
        mode=normalized_mode,
        message=message,
        reason=reason,
        title=title,
        source_node_id=source_node_id,
    )


def _optional_mapping_list(
    payload: Iterable[Mapping[str, Any] | dict[str, Any]] | None,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    if payload is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} entries must be mappings")
        normalized.append(dict(item))
    return normalized


def _require_smr_credential_provider(
    value: SmrCredentialProvider | str | None,
    *,
    field_name: str = "provider",
) -> SmrCredentialProvider:
    provider = coerce_smr_credential_provider(value, field_name=field_name)
    if provider is None:
        raise ValueError(f"{field_name} is required")
    return provider


def _require_smr_funding_source(
    value: SmrFundingSource | str | None,
    *,
    field_name: str = "funding_source",
) -> SmrFundingSource:
    funding_source = coerce_smr_funding_source(value, field_name=field_name)
    if funding_source is None:
        raise ValueError(f"{field_name} is required")
    return funding_source


def _normalized_actor_model_assignment_payloads(
    payload: Iterable[SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]] | None,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    normalized = normalize_actor_model_assignments(
        list(payload) if payload is not None else None,
        field_name=field_name,
    )
    return [item.as_payload() for item in normalized]


def _runnable_project_request_payload(
    request: SmrRunnableProjectRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, SmrRunnableProjectRequest):
        return request.to_wire()
    if isinstance(request, Mapping):
        return SmrRunnableProjectRequest.from_wire(dict(request)).to_wire()
    raise ValueError("request must be a SmrRunnableProjectRequest or mapping")


def _merge_execution_actor_model_assignments(
    payload: Mapping[str, Any] | dict[str, Any],
    *,
    actor_model_assignments: Iterable[SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]]
    | None,
) -> dict[str, Any]:
    normalized_payload = dict(payload)
    assignment_payloads = _normalized_actor_model_assignment_payloads(
        actor_model_assignments,
        field_name="actor_model_assignments",
    )
    if not assignment_payloads:
        return normalized_payload
    raw_execution = normalized_payload.get("execution")
    execution: dict[str, Any] = dict(raw_execution) if isinstance(raw_execution, Mapping) else {}
    if any(
        execution.get(key)
        for key in (
            "agent_kind",
            "agent_model",
            "agent_model_params",
            "agent_profiles",
        )
    ):
        raise ValueError(
            "actor_model_assignments cannot be combined with shared execution.agent_kind/"
            "agent_model/agent_model_params/agent_profiles"
        )
    execution["actor_model_assignments"] = assignment_payloads
    normalized_payload["execution"] = execution
    return normalized_payload


def _normalized_roles_payload(
    roles: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    normalized = coerce_smr_role_bindings(roles, field_name=field_name)
    if normalized is None:
        return None
    return normalized.to_wire()


def _primary_objective_ref_payload(
    *,
    primary_objective_id: str | None,
    primary_objective_kind: str | None,
    primary_parent_ref: Mapping[str, Any] | dict[str, Any] | None,
    primary_parent: Mapping[str, Any] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    objective_id = str(primary_objective_id or "").strip()
    objective_kind = str(primary_objective_kind or "").strip()
    if not objective_id:
        return None
    if primary_parent_ref is not None or primary_parent is not None:
        raise ValueError(
            "primary_objective_id cannot be combined with primary_parent_ref or primary_parent"
        )
    payload: dict[str, Any] = {"id": objective_id}
    if objective_kind:
        payload["kind"] = objective_kind
    return payload


def _build_project_run_payload(
    *,
    host_kind: SmrHostKind | str | None = None,
    work_mode: SmrWorkMode | str | None = None,
    mode: SmrWorkMode | str | None = None,
    intended_horizon_hours: SmrIntendedHorizonHours | int | None = None,
    providers: Iterable[ProviderBinding | str | Mapping[str, Any] | dict[str, Any]] | None = None,
    limit: UsageLimit | Mapping[str, Any] | dict[str, Any] | None = None,
    worker_pool_id: str | None = None,
    runbook: SmrRunbookKind | str | None = None,
    runbook_preset: str | None = None,
    runbook_config_id: str | None = None,
    local_execution: Mapping[str, Any] | dict[str, Any] | None = None,
    execution_profile: LocalExecutionProfile | Mapping[str, Any] | dict[str, Any] | None = None,
    timebox_seconds: int | None = None,
    agent_profile: str | None = None,
    agent_model: SmrAgentModel | str | None = None,
    agent_harness: SmrAgentHarness | None = None,
    agent_kind: SmrAgentKind | None = None,
    agent_model_params: Mapping[str, Any] | dict[str, Any] | None = None,
    actor_model_overrides: Iterable[SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]]
    | None = None,
    roles: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None = None,
    initial_runtime_messages: Iterable[Mapping[str, Any] | dict[str, Any]] | None = None,
    workflow: Mapping[str, Any] | dict[str, Any] | None = None,
    sandbox_override: Mapping[str, Any] | dict[str, Any] | None = None,
    environment: Mapping[str, Any] | dict[str, Any] | None = None,
    run_policy: SmrRunPolicy | Mapping[str, Any] | dict[str, Any] | None = None,
    kickoff_contract: KickoffContract | Mapping[str, Any] | dict[str, Any] | None = None,
    resource_bindings: RunResourceBindings | Mapping[str, Any] | dict[str, Any] | None = None,
    ai_cache: Mapping[str, Any] | dict[str, Any] | None = None,
    primary_objective_id: str | None = None,
    primary_objective_kind: str | None = None,
    primary_parent_ref: Mapping[str, Any] | dict[str, Any] | None = None,
    primary_parent: Mapping[str, Any] | dict[str, Any] | None = None,
    effort_id: str | None = None,
    idempotency_key_run_create: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    preset_id = _optional_non_empty_string(runbook_preset)
    config_id = _optional_non_empty_string(runbook_config_id)
    if preset_id and config_id and preset_id != config_id:
        raise ValueError("runbook_preset and runbook_config_id must match when both are provided")
    normalized_horizon = coerce_intended_horizon_hours(
        intended_horizon_hours,
        field_name="intended_horizon_hours",
    )
    uses_backend_profile = bool(preset_id or config_id or normalized_horizon is not None)
    payload: dict[str, Any] = {}
    if preset_id:
        payload["runbook_preset"] = preset_id
    if config_id:
        payload["runbook_config_id"] = config_id

    normalized_runbook = coerce_smr_runbook_kind(runbook, field_name="runbook")
    if normalized_runbook is not None:
        payload["runbook"] = normalized_runbook.value

    normalized_host_kind = coerce_smr_host_kind(host_kind, field_name="host_kind")
    if normalized_host_kind is not None:
        payload["host_kind"] = normalized_host_kind.value
    elif not uses_backend_profile:
        raise ValueError(
            "host_kind is required unless runbook_preset or intended_horizon_hours is provided"
        )

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
    elif not uses_backend_profile:
        raise ValueError(
            "work_mode is required unless runbook_preset or intended_horizon_hours is provided"
        )

    provider_values = list(providers) if providers is not None else None
    if provider_values is not None:
        payload["providers"] = [
            binding.to_wire()
            for binding in coerce_provider_bindings(
                provider_values,
                field_name="providers",
            )
        ]
    elif not uses_backend_profile:
        raise ValueError(
            "providers is required unless runbook_preset or intended_horizon_hours is provided"
        )

    normalized_limit = coerce_usage_limit(limit, field_name="limit")
    if normalized_limit is not None:
        limit_payload = normalized_limit.to_dict()
        if limit_payload:
            payload["limit"] = limit_payload
    normalized_actor_model_overrides = _normalized_actor_model_assignment_payloads(
        actor_model_overrides,
        field_name="actor_model_overrides",
    )
    normalized_roles = _normalized_roles_payload(roles, field_name="roles")
    if normalized_roles and normalized_actor_model_overrides:
        raise ValueError("roles cannot be combined with actor_model_overrides")
    if normalized_roles and (
        (agent_profile and agent_profile.strip())
        or agent_model is not None
        or agent_harness is not None
        or agent_kind is not None
        or agent_model_params is not None
    ):
        raise ValueError(
            "roles cannot be combined with shared top-level "
            "agent_profile/agent_model/agent_harness/agent_kind/agent_model_params"
        )
    if worker_pool_id and worker_pool_id.strip():
        payload["worker_pool_id"] = worker_pool_id.strip()
    normalized_local_execution = _optional_mapping(
        local_execution,
        field_name="local_execution",
    )
    if normalized_local_execution:
        payload["local_execution"] = normalized_local_execution
    if execution_profile is not None:
        if isinstance(execution_profile, LocalExecutionProfile):
            payload["execution_profile"] = execution_profile.to_request_wire()
        else:
            payload["execution_profile"] = _optional_mapping(
                execution_profile,
                field_name="execution_profile",
            )
    if normalized_horizon is not None:
        payload["intended_horizon_hours"] = int(normalized_horizon)
        if timebox_seconds is not None and int(timebox_seconds) != int(normalized_horizon) * 3600:
            raise ValueError(
                "timebox_seconds must equal intended_horizon_hours * 3600 when both are provided"
            )
    if timebox_seconds is not None:
        payload["timebox_seconds"] = int(timebox_seconds)
    if agent_profile and agent_profile.strip():
        payload["agent_profile"] = agent_profile.strip()
    if normalized_actor_model_overrides and (
        (agent_profile and agent_profile.strip())
        or agent_model is not None
        or agent_harness is not None
        or agent_kind is not None
        or agent_model_params is not None
    ):
        raise ValueError(
            "actor_model_overrides cannot be combined with shared top-level "
            "agent_profile/agent_model/agent_harness/agent_kind/agent_model_params"
        )
    normalized_agent_model = validate_shared_top_level_agent_model(
        agent_model,
        field_name="agent_model",
    )
    if normalized_agent_model is not None:
        payload["agent_model"] = normalized_agent_model.value
    normalized_agent_harness = coerce_smr_agent_harness(
        agent_harness,
        field_name="agent_harness",
    )
    normalized_agent_kind = coerce_smr_agent_kind(agent_kind, field_name="agent_kind")
    if (
        normalized_agent_harness is not None
        and normalized_agent_kind is not None
        and normalized_agent_harness.value != normalized_agent_kind.value
    ):
        raise ValueError("agent_harness and agent_kind must match when both are provided")
    resolved_agent_harness = normalized_agent_harness or normalized_agent_kind
    if resolved_agent_harness is not None:
        payload["agent_harness"] = resolved_agent_harness.value
    normalized_agent_model_params = _optional_mapping(
        agent_model_params,
        field_name="agent_model_params",
    )
    if normalized_agent_model_params:
        payload["agent_model_params"] = normalized_agent_model_params
    if normalized_actor_model_overrides:
        payload["actor_model_overrides"] = normalized_actor_model_overrides
    if normalized_roles:
        payload["roles"] = normalized_roles
    normalized_initial_runtime_messages = _optional_mapping_list(
        initial_runtime_messages,
        field_name="initial_runtime_messages",
    )
    if normalized_initial_runtime_messages:
        payload["initial_runtime_messages"] = normalized_initial_runtime_messages
    normalized_workflow = _optional_mapping(workflow, field_name="workflow")
    if normalized_workflow:
        payload["workflow"] = normalized_workflow
    normalized_sandbox_override = _optional_mapping(
        sandbox_override,
        field_name="sandbox_override",
    )
    if normalized_sandbox_override:
        payload["sandbox_override"] = normalized_sandbox_override
    normalized_environment = _optional_mapping(
        environment,
        field_name="environment",
    )
    if normalized_environment:
        payload["environment"] = normalized_environment
    normalized_run_policy = coerce_smr_run_policy(run_policy, field_name="run_policy")
    if normalized_run_policy is not None:
        payload["run_policy"] = normalized_run_policy.to_dict()
    if kickoff_contract is not None:
        if isinstance(kickoff_contract, KickoffContract):
            payload["kickoff_contract"] = kickoff_contract.to_wire()
        else:
            payload["kickoff_contract"] = KickoffContract.from_wire(
                _optional_mapping(
                    kickoff_contract,
                    field_name="kickoff_contract",
                )
            ).to_wire()
    if resource_bindings is not None:
        if isinstance(resource_bindings, RunResourceBindings):
            normalized_resource_bindings = resource_bindings.to_wire()
        else:
            normalized_resource_bindings = _optional_mapping(
                resource_bindings,
                field_name="resource_bindings",
            )
        if normalized_resource_bindings:
            payload["resource_bindings"] = normalized_resource_bindings
    normalized_ai_cache = _optional_mapping(ai_cache, field_name="ai_cache")
    if normalized_ai_cache:
        payload["ai_cache"] = normalized_ai_cache
    primary_objective_ref = _primary_objective_ref_payload(
        primary_objective_id=primary_objective_id,
        primary_objective_kind=primary_objective_kind,
        primary_parent_ref=primary_parent_ref,
        primary_parent=primary_parent,
    )
    normalized_primary_parent_ref = _optional_mapping(
        primary_objective_ref or primary_parent_ref,
        field_name="primary_parent_ref",
    )
    if normalized_primary_parent_ref:
        payload["primary_parent_ref"] = normalized_primary_parent_ref
    normalized_primary_parent = _optional_mapping(
        primary_parent,
        field_name="primary_parent",
    )
    if normalized_primary_parent:
        payload["primary_parent"] = normalized_primary_parent
    if effort_id and effort_id.strip():
        payload["effort_id"] = effort_id.strip()
    if idempotency_key_run_create and idempotency_key_run_create.strip():
        payload["idempotency_key_run_create"] = idempotency_key_run_create.strip()
    if idempotency_key and idempotency_key.strip():
        payload["idempotency_key"] = idempotency_key.strip()
    return payload


def _guess_content_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _is_source_bundle_entry(path: str, entry: Mapping[str, Any]) -> bool:
    kind = str(entry.get("kind") or "").strip().lower()
    content_type = str(entry.get("content_type") or _guess_content_type(path)).strip().lower()
    return (
        kind == "source_bundle"
        or path.lower().endswith(".zip")
        or content_type
        in {
            "application/zip",
            "application/x-zip",
            "application/x-zip-compressed",
            "multipart/x-zip",
        }
    )


def _normalize_uploaded_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = str(entry.get("path") or "").strip()
    if not path:
        raise ValueError("workspace file entries require a non-empty path")
    content = entry.get("content")
    content_path = entry.get("content_path")
    content_type = str(entry.get("content_type") or _guess_content_type(path)).strip()
    encoding = str(entry.get("encoding") or "").strip().lower() or None
    if content_path is not None:
        file_path = Path(str(content_path))
        raw_bytes = file_path.read_bytes()
        if _is_source_bundle_entry(path, {**dict(entry), "content_type": content_type}):
            content = base64.b64encode(raw_bytes).decode("ascii")
            encoding = encoding or "base64"
        else:
            try:
                content = raw_bytes.decode("utf-8")
                encoding = encoding or "utf-8"
            except UnicodeDecodeError:
                content = base64.b64encode(raw_bytes).decode("ascii")
                encoding = encoding or "base64"
    if content is None:
        raise ValueError("workspace file entries require either content or content_path")
    if isinstance(content, bytes):
        content = base64.b64encode(content).decode("ascii")
        encoding = encoding or "base64"
    if not isinstance(content, str):
        raise ValueError("workspace file content must be text or bytes")
    normalized: dict[str, Any] = {
        "path": path,
        "content": content,
        "content_type": content_type,
        "encoding": encoding or "utf-8",
    }
    kind = str(entry.get("kind") or "").strip()
    if kind:
        normalized["kind"] = kind
    metadata = entry.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("uploaded file metadata must be a mapping when provided")
        normalized["metadata"] = dict(metadata)
    return normalized


def _normalize_resource_uploaded_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_uploaded_file(entry)
    visibility = str(entry.get("visibility") or "").strip()
    if visibility:
        normalized["visibility"] = visibility
    return normalized


def _source_bundle_file_entry(
    bundle_path: str | os.PathLike[str],
    *,
    path: str | None = None,
    visibility: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(bundle_path)
    upload_path = str(path or resolved.name).strip()
    if not upload_path:
        raise ValueError("source bundle upload path must be non-empty")
    payload: dict[str, Any] = {
        "path": upload_path,
        "content_path": resolved,
        "content_type": "application/zip",
        "kind": "source_bundle",
        "metadata": {
            "kind": "source_bundle",
            **dict(metadata or {}),
        },
    }
    if visibility:
        payload["visibility"] = visibility
    return payload


@dataclass
class ManagedResearchClient:
    """Managed Research client implementation."""

    api_key: str | None = None
    backend_base: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _transport: SmrHttpTransport = field(init=False, repr=False)
    _projects_api: ProjectsAPI | None = field(init=False, default=None, repr=False)
    _factories_api: FactoriesAPI | None = field(init=False, default=None, repr=False)
    _efforts_api: EffortsAPI | None = field(init=False, default=None, repr=False)
    _runs_api: RunsAPI | None = field(init=False, default=None, repr=False)
    _workspace_inputs_api: WorkspaceInputsAPI | None = field(init=False, default=None, repr=False)
    _progress_api: ProgressAPI | None = field(init=False, default=None, repr=False)
    _setup_api: SetupAPI | None = field(init=False, default=None, repr=False)
    _approvals_api: ApprovalsAPI | None = field(init=False, default=None, repr=False)
    _files_api: FilesAPI | None = field(init=False, default=None, repr=False)
    _github_api: GithubAPI | None = field(init=False, default=None, repr=False)
    _exports_api: ExportsAPI | None = field(init=False, default=None, repr=False)
    _outputs_api: OutputsAPI | None = field(init=False, default=None, repr=False)
    _prs_api: PrsAPI | None = field(init=False, default=None, repr=False)
    _readiness_api: ReadinessAPI | None = field(init=False, default=None, repr=False)
    _repos_api: ReposAPI | None = field(init=False, default=None, repr=False)
    _datasets_api: DatasetsAPI | None = field(init=False, default=None, repr=False)
    _models_api: ModelsAPI | None = field(init=False, default=None, repr=False)
    _repositories_api: RepositoriesAPI | None = field(init=False, default=None, repr=False)
    _credentials_api: CredentialsAPI | None = field(init=False, default=None, repr=False)
    _secrets_api: SecretsAPI | None = field(init=False, default=None, repr=False)
    _environments_api: EnvironmentsAPI | None = field(init=False, default=None, repr=False)
    _logs_api: LogsAPI | None = field(init=False, default=None, repr=False)
    _integrations_api: IntegrationsAPI | None = field(init=False, default=None, repr=False)
    _usage_api: UsageAPI | None = field(init=False, default=None, repr=False)
    _trained_models_api: TrainedModelsAPI | None = field(init=False, default=None, repr=False)
    _run_cost_api: RunCostAPI | None = field(init=False, default=None, repr=False)
    _work_products_api: WorkProductsAPI | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        resolved_api_key = _resolve_api_key(self.api_key)
        resolved_backend_base = _resolve_backend_base(self.backend_base)
        self.api_key = resolved_api_key
        self.backend_base = resolved_backend_base
        self._transport = build_http_transport(
            api_key=resolved_api_key,
            backend_base=resolved_backend_base,
            timeout_seconds=self.timeout_seconds,
        )

    def __enter__(self) -> ManagedResearchClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._transport.close()

    @property
    def projects(self) -> ProjectsAPI:
        if self._projects_api is None:
            self._projects_api = ProjectsAPI(self)
        return self._projects_api

    @property
    def factories(self) -> FactoriesAPI:
        if self._factories_api is None:
            self._factories_api = FactoriesAPI(self)
        return self._factories_api

    @property
    def efforts(self) -> EffortsAPI:
        if self._efforts_api is None:
            self._efforts_api = EffortsAPI(self)
        return self._efforts_api

    @property
    def runs(self) -> RunsAPI:
        if self._runs_api is None:
            self._runs_api = RunsAPI(self)
        return self._runs_api

    def run(self, project_id: str, run_id: str):
        from synth_ai.managed_research.sdk.runs import RunHandle

        return RunHandle(self, project_id, run_id)

    @property
    def workspace_inputs(self) -> WorkspaceInputsAPI:
        if self._workspace_inputs_api is None:
            self._workspace_inputs_api = WorkspaceInputsAPI(self)
        return self._workspace_inputs_api

    @property
    def progress(self) -> ProgressAPI:
        if self._progress_api is None:
            self._progress_api = ProgressAPI(self)
        return self._progress_api

    @property
    def setup(self) -> SetupAPI:
        if self._setup_api is None:
            self._setup_api = SetupAPI(self)
        return self._setup_api

    @property
    def approvals(self) -> ApprovalsAPI:
        if self._approvals_api is None:
            self._approvals_api = ApprovalsAPI(self)
        return self._approvals_api

    @property
    def files(self) -> FilesAPI:
        if self._files_api is None:
            self._files_api = FilesAPI(self)
        return self._files_api

    @property
    def github(self) -> GithubAPI:
        if self._github_api is None:
            self._github_api = GithubAPI(self)
        return self._github_api

    @property
    def exports(self) -> ExportsAPI:
        if self._exports_api is None:
            self._exports_api = ExportsAPI(self)
        return self._exports_api

    @property
    def outputs(self) -> OutputsAPI:
        if self._outputs_api is None:
            self._outputs_api = OutputsAPI(self)
        return self._outputs_api

    @property
    def prs(self) -> PrsAPI:
        if self._prs_api is None:
            self._prs_api = PrsAPI(self)
        return self._prs_api

    @property
    def readiness(self) -> ReadinessAPI:
        if self._readiness_api is None:
            self._readiness_api = ReadinessAPI(self)
        return self._readiness_api

    @property
    def repos(self) -> ReposAPI:
        if self._repos_api is None:
            self._repos_api = ReposAPI(self)
        return self._repos_api

    @property
    def datasets(self) -> DatasetsAPI:
        if self._datasets_api is None:
            self._datasets_api = DatasetsAPI(self)
        return self._datasets_api

    @property
    def models(self) -> ModelsAPI:
        if self._models_api is None:
            self._models_api = ModelsAPI(self)
        return self._models_api

    @property
    def repositories(self) -> RepositoriesAPI:
        if self._repositories_api is None:
            self._repositories_api = RepositoriesAPI(self)
        return self._repositories_api

    @property
    def credentials(self) -> CredentialsAPI:
        warnings.warn(
            "client.credentials is deprecated; use client.secrets instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._credentials_api is None:
            self._credentials_api = CredentialsAPI(self)
        return self._credentials_api

    @property
    def secrets(self) -> SecretsAPI:
        if self._secrets_api is None:
            self._secrets_api = SecretsAPI(self)
        return self._secrets_api

    @property
    def environments(self) -> EnvironmentsAPI:
        if self._environments_api is None:
            self._environments_api = EnvironmentsAPI(self)
        return self._environments_api

    @property
    def logs(self) -> LogsAPI:
        if self._logs_api is None:
            self._logs_api = LogsAPI(self)
        return self._logs_api

    @property
    def integrations(self) -> IntegrationsAPI:
        if self._integrations_api is None:
            self._integrations_api = IntegrationsAPI(self)
        return self._integrations_api

    def project(self, project_id: str) -> ManagedResearchProjectClient:
        return ManagedResearchProjectClient(
            self,
            _require_non_empty_string(project_id, field_name="project_id"),
        )

    @property
    def usage(self) -> UsageAPI:
        if self._usage_api is None:
            self._usage_api = UsageAPI(self)
        return self._usage_api

    @property
    def trained_models(self) -> TrainedModelsAPI:
        if self._trained_models_api is None:
            self._trained_models_api = TrainedModelsAPI(self)
        return self._trained_models_api

    @property
    def work_products(self) -> WorkProductsAPI:
        if self._work_products_api is None:
            self._work_products_api = WorkProductsAPI(self)
        return self._work_products_api

    @property
    def run_cost(self) -> RunCostAPI:
        if self._run_cost_api is None:
            self._run_cost_api = RunCostAPI(self)
        return self._run_cost_api

    def get_billing_entitlements(self) -> BillingEntitlementSnapshot:
        return self.usage.get_billing_entitlements()

    def get_run_usage(self, run_id: str) -> SmrRunUsage:
        return self.usage.get_run_usage(run_id)

    def get_run_resource_limits(self, run_id: str) -> SmrResourceLimits:
        return self.usage.get_run_resource_limits(run_id)

    def get_run_progress_toward_resource_limits(
        self,
        run_id: str,
    ) -> SmrResourceLimitProgress:
        return self.usage.get_run_progress_toward_resource_limits(run_id)

    def extend_run_resource_limit(
        self,
        run_id: str,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return self.usage.extend_run_resource_limit(
            run_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def get_project_run_resource_limits(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrResourceLimits:
        return self.usage.get_project_run_resource_limits(project_id, run_id)

    def get_project_run_progress_toward_resource_limits(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrResourceLimitProgress:
        return self.usage.get_project_run_progress_toward_resource_limits(
            project_id,
            run_id,
        )

    def extend_project_run_resource_limit(
        self,
        project_id: str,
        run_id: str,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return self.usage.extend_project_run_resource_limit(
            project_id,
            run_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def get_project_usage(self, project_id: str) -> SmrProjectUsage:
        return self.usage.get_project_usage(project_id)

    def get_project_resource_limits(self, project_id: str) -> SmrResourceLimits:
        return self.usage.get_project_resource_limits(project_id)

    def get_project_progress_toward_resource_limits(
        self,
        project_id: str,
    ) -> SmrResourceLimitProgress:
        return self.usage.get_project_progress_toward_resource_limits(project_id)

    def extend_project_resource_limit(
        self,
        project_id: str,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return self.usage.extend_project_resource_limit(
            project_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def get_project_economics(self, project_id: str) -> SmrProjectEconomics:
        return self.usage.get_project_economics(project_id)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        return self._transport.request_json(
            method,
            path,
            params=params,
            json_body=json_body,
            allow_not_found=allow_not_found,
        )

    def _stream_sse(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        last_event_id: str | None = None,
        timeout: float | None = None,
    ):
        return self._transport.stream_sse(
            path,
            params=params,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def get_backend_version(self) -> dict[str, Any]:
        """Return backend SemVer and deploy metadata."""
        return _coerce_dict(
            self._request_json("GET", "/api/v1/version"),
            label="get_backend_version",
        )

    def _request_content(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._transport.client.request(
            method,
            path,
            params=params,
            follow_redirects=True,
        )
        if response.is_error:
            _raise_for_error_response(response)
        raw_bytes = response.content
        content_type = str(response.headers.get("content-type") or "").strip() or None
        try:
            content_text = raw_bytes.decode("utf-8")
            return {
                "content": content_text,
                "encoding": "utf-8",
                "content_type": content_type,
                "content_bytes": len(raw_bytes),
            }
        except UnicodeDecodeError:
            return {
                "content": base64.b64encode(raw_bytes).decode("ascii"),
                "encoding": "base64",
                "content_type": content_type,
                "content_bytes": len(raw_bytes),
            }

    def create_runnable_project(
        self,
        request: SmrRunnableProjectRequest | Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                "/smr/projects:runnable",
                json_body=_runnable_project_request_payload(request),
            ),
            label="create_runnable_project",
        )

    def list_projects(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params = build_query_params(
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
        )
        return _coerce_dict_list(
            self._request_json("GET", "/smr/projects", params=params), label="list_projects"
        )

    def get_project(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}"), label="get_project"
        )

    def create_factory(
        self,
        request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                "/smr/factories",
                json_body=factory_create_payload(request),
            ),
            label="create_factory",
        )

    def list_factories(self) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", "/smr/factories"),
            label="list_factories",
        )

    def get_factory(self, factory_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/factories/{factory_id}"),
            label="get_factory",
        )

    def patch_factory(
        self,
        factory_id: str,
        request: FactoryPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/factories/{factory_id}",
                json_body=factory_patch_payload(request),
            ),
            label="patch_factory",
        )

    def get_factory_status(self, factory_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/factories/{factory_id}/status"),
            label="get_factory_status",
        )

    def list_efforts_for_factory(self, factory_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/factories/{factory_id}/efforts"),
            label="list_efforts_for_factory",
        )

    def create_effort(
        self,
        request: EffortCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                "/smr/efforts",
                json_body=effort_create_payload(request),
            ),
            label="create_effort",
        )

    def get_effort(self, effort_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/efforts/{effort_id}"),
            label="get_effort",
        )

    def patch_effort(
        self,
        effort_id: str,
        request: EffortPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/efforts/{effort_id}",
                json_body=effort_patch_payload(request),
            ),
            label="patch_effort",
        )

    def get_default_project(self) -> dict[str, Any]:
        return self.get_project(DEFAULT_MISC_PROJECT_ALIAS)

    def update_project_schedule(
        self,
        project_id: str,
        schedule: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/schedule",
                json_body={"schedule": dict(schedule)},
            ),
            label="update_project_schedule",
        )

    def patch_project(
        self,
        project_id: str,
        payload: dict[str, Any],
        *,
        actor_model_assignments: Iterable[
            SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]
        ]
        | None = None,
    ) -> dict[str, Any]:
        normalized_payload = _merge_execution_actor_model_assignments(
            payload,
            actor_model_assignments=actor_model_assignments,
        )
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}",
                json_body=normalized_payload,
            ),
            label="patch_project",
        )

    def rename_project(self, project_id: str, name: str) -> dict[str, Any]:
        next_name = str(name or "").strip()
        if not next_name:
            raise ValueError("project name must be non-empty")
        return self.patch_project(project_id, {"name": next_name})

    def pause_project(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/pause"),
            label="pause_project",
        )

    def resume_project(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/resume"),
            label="resume_project",
        )

    def archive_project(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/archive"),
            label="archive_project",
        )

    def unarchive_project(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/unarchive"),
            label="unarchive_project",
        )

    def get_project_notes(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/notes"),
            label="get_project_notes",
        )

    def set_project_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PUT",
                f"/smr/projects/{project_id}/notes",
                json_body={"notes": str(notes)},
            ),
            label="set_project_notes",
        )

    def append_project_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/notes/append",
                json_body={"text": str(notes)},
            ),
            label="append_project_notes",
        )

    def get_org_knowledge(self) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", "/smr/org/knowledge"),
            label="get_org_knowledge",
        )

    def set_org_knowledge(self, content: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PUT",
                "/smr/org/knowledge",
                json_body={"content": str(content)},
            ),
            label="set_org_knowledge",
        )

    def get_project_knowledge(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/knowledge"),
            label="get_project_knowledge",
        )

    def set_project_knowledge(self, project_id: str, content: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PUT",
                f"/smr/projects/{project_id}/knowledge",
                json_body={"content": str(content)},
            ),
            label="set_project_knowledge",
        )

    def get_project_status(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/status"),
            label="get_project_status",
        )

    def get_project_workspace(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/workspace"),
            label="get_project_workspace",
        )

    def list_project_changesets(
        self,
        project_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/changesets",
                params=build_query_params(status=status, limit=limit),
            ),
            label="list_project_changesets",
        )

    def create_project_changeset(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/changesets",
                json_body=dict(payload),
            ),
            label="create_project_changeset",
        )

    def get_project_changeset(
        self,
        project_id: str,
        changeset_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/changesets/{changeset_id}",
            ),
            label="get_project_changeset",
        )

    def decide_project_changeset(
        self,
        project_id: str,
        changeset_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/changesets/{changeset_id}/decision",
                json_body=dict(payload),
            ),
            label="decide_project_changeset",
        )

    def get_project_status_snapshot(self, project_id: str) -> dict[str, Any]:
        return self.get_project_status(project_id)

    def get_project_readiness(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/readiness"),
            label="get_project_readiness",
        )

    def get_project_entitlement(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/entitlements/managed_research",
            ),
            label="get_project_entitlement",
        )

    def get_capabilities(self) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", "/smr/capabilities"), label="get_capabilities"
        )

    def get_agent_models(self) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", "/smr/agent-models"),
            label="get_agent_models",
        )

    def get_limits(self) -> dict[str, Any]:
        return _coerce_dict(self._request_json("GET", "/smr/limits"), label="get_limits")

    def get_capacity_lane_preview(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/capacity-lane-preview"),
            label="get_capacity_lane_preview",
        )

    def get_workspace_download_url(self, project_id: str) -> dict[str, Any]:
        """Return a presigned URL and metadata for the project workspace tarball."""
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/workspace/download-url",
            ),
            label="get_workspace_download_url",
        )

    def get_project_git(self, project_id: str) -> dict[str, Any]:
        """Return read-only git metadata for the project workspace (commit, branch, remote hints)."""
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/git"),
            label="get_project_git",
        )

    def download_workspace_archive(
        self,
        project_id: str,
        output_path: str | os.PathLike[str],
        *,
        timeout_seconds: float = DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Fetch the workspace tarball via the presigned URL and write it to *output_path*."""
        info = self.get_workspace_download_url(project_id)
        url = info.get("download_url")
        if not isinstance(url, str) or not url.strip():
            raise SmrApiError(
                "Workspace download response missing download_url",
                status_code=None,
                response_text=None,
            )
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                httpx.Client(timeout=timeout_seconds) as http_client,
                http_client.stream("GET", url.strip()) as response,
                path.open("wb") as file_handle,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    file_handle.write(chunk)
        except httpx.HTTPError as exc:
            raise SmrApiError(
                f"Failed to download workspace archive: {exc}",
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
                response_text=getattr(getattr(exc, "response", None), "text", None),
            ) from exc
        return {
            "output_path": str(path),
            "commit_sha": info.get("commit_sha"),
            "archive_key": info.get("archive_key"),
            "bytes_written": path.stat().st_size,
        }

    def download_run_workspace_archive(
        self,
        project_id: str,
        run_id: str,
        output_path: str | os.PathLike[str],
        *,
        timeout_seconds: float = DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Download the immutable archive resolved for one run."""
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        request_path = f"/smr/projects/{project_id}/runs/{run_id}/workspace/archive"
        try:
            with (
                self._transport.client.stream(
                    "GET",
                    request_path,
                    timeout=timeout_seconds,
                ) as response,
                path.open("wb") as file_handle,
            ):
                if response.is_error:
                    response.read()
                    _raise_for_error_response(response)
                for chunk in response.iter_bytes():
                    file_handle.write(chunk)
                commit_sha = response.headers.get("x-workspace-commit")
                archive_key = response.headers.get("x-workspace-archive-key")
        except httpx.HTTPError as exc:
            raise SmrApiError(
                f"Failed to download run workspace archive: {exc}",
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
                response_text=getattr(getattr(exc, "response", None), "text", None),
            ) from exc
        return {
            "output_path": str(path),
            "project_id": project_id,
            "run_id": run_id,
            "commit_sha": commit_sha,
            "archive_key": archive_key,
            "bytes_written": path.stat().st_size,
        }

    def attach_source_repo(
        self,
        project_id: str,
        url: str,
        *,
        default_branch: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": str(url).strip()}
        if default_branch and default_branch.strip():
            payload["default_branch"] = default_branch.strip()
        return _coerce_dict(
            self._request_json(
                "PUT",
                f"/smr/projects/{project_id}/workspace-inputs/source-repo",
                json_body=payload,
            ),
            label="attach_source_repo",
        )

    def get_workspace_inputs(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/workspace-inputs"),
            label="get_workspace_inputs",
        )

    def upload_workspace_files(
        self,
        project_id: str,
        files: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_files = [_normalize_uploaded_file(entry) for entry in files]
        if not normalized_files:
            raise ValueError("upload_workspace_files requires at least one file")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/workspace-inputs/files:upload",
                json_body={"files": normalized_files},
            ),
            label="upload_workspace_files",
        )

    def upload_workspace_directory(
        self,
        project_id: str,
        directory: str | os.PathLike[str],
    ) -> dict[str, Any]:
        root = Path(directory).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace directory does not exist: {root}")
        files: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "content_path": path,
                    "content_type": _guess_content_type(path.name),
                }
            )
        return self.upload_workspace_files(project_id, files)

    def upload_workspace_source_bundle(
        self,
        project_id: str,
        bundle_path: str | os.PathLike[str],
        *,
        path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.upload_workspace_files(
            project_id,
            [
                _source_bundle_file_entry(
                    bundle_path,
                    path=path,
                    metadata=metadata,
                )
            ],
        )

    def list_project_files(
        self,
        project_id: str,
        *,
        visibility: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/files",
                params=build_query_params(visibility=visibility, limit=limit),
            ),
            label="list_project_files",
        )

    def create_project_files(
        self,
        project_id: str,
        files: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_files = [_normalize_resource_uploaded_file(entry) for entry in files]
        if not normalized_files:
            raise ValueError("create_project_files requires at least one file")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/files",
                json_body={"files": normalized_files},
            ),
            label="create_project_files",
        )

    def create_project_source_bundle(
        self,
        project_id: str,
        bundle_path: str | os.PathLike[str],
        *,
        path: str | None = None,
        visibility: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.create_project_files(
            project_id,
            [
                _source_bundle_file_entry(
                    bundle_path,
                    path=path,
                    visibility=visibility,
                    metadata=metadata,
                )
            ],
        )

    def get_project_file(self, project_id: str, file_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/files/{file_id}"),
            label="get_project_file",
        )

    def get_project_file_content(self, project_id: str, file_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content(
                "GET",
                f"/smr/projects/{project_id}/files/{file_id}/content",
            ),
            label="get_project_file_content",
        )

    def list_project_datasets(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/datasets"),
            label="list_project_datasets",
        )

    def upload_project_dataset(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/datasets",
                json_body=dict(payload),
            ),
            label="upload_project_dataset",
        )

    def download_project_dataset(self, project_id: str, dataset_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content(
                "GET",
                f"/smr/projects/{project_id}/datasets/{dataset_id}/download",
            ),
            label="download_project_dataset",
        )

    def get_file_content(self, file_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content("GET", f"/smr/files/{file_id}/content"),
            label="get_file_content",
        )

    def list_run_file_mounts(self, run_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/file-mounts"),
            label="list_run_file_mounts",
        )

    def upload_run_files(
        self,
        run_id: str,
        files: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_files = [_normalize_resource_uploaded_file(entry) for entry in files]
        if not normalized_files:
            raise ValueError("upload_run_files requires at least one file")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/files:upload",
                json_body={"files": normalized_files},
            ),
            label="upload_run_files",
        )

    def upload_run_source_bundle(
        self,
        run_id: str,
        bundle_path: str | os.PathLike[str],
        *,
        path: str | None = None,
        visibility: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.upload_run_files(
            run_id,
            [
                _source_bundle_file_entry(
                    bundle_path,
                    path=path,
                    visibility=visibility,
                    metadata=metadata,
                )
            ],
        )

    def _list_run_output_files(
        self,
        run_id: str,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/outputs",
                params=build_query_params(artifact_type=artifact_type, limit=limit),
            ),
            label="list_run_output_files",
        )

    def _list_project_run_artifacts(
        self,
        project_id: str,
        run_id: str,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            artifact.__dict__
            for artifact in self.list_run_artifacts(
                run_id,
                project_id=project_id,
                artifact_type=artifact_type,
                limit=limit,
                cursor=cursor,
            )
        ]

    def list_run_artifacts(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[RunArtifact]:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/artifacts"
            if project_id
            else f"/smr/runs/{run_id}/artifacts"
        )
        payload = _coerce_dict_list(
            self._request_json(
                "GET",
                path,
                params=build_query_params(
                    artifact_type=artifact_type,
                    limit=limit,
                    cursor=cursor,
                ),
            ),
            label="list_run_artifacts",
        )
        return [RunArtifact.from_wire(item) for item in payload]

    def get_run_artifact_manifest(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> RunArtifactManifest:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/artifacts/manifest"
            if project_id
            else f"/smr/runs/{run_id}/artifacts/manifest"
        )
        return RunArtifactManifest.from_wire(
            _coerce_dict(
                self._request_json("GET", path),
                label="get_run_artifact_manifest",
            )
        )

    def get_artifact(self, artifact_id: str) -> RunArtifact:
        return RunArtifact.from_wire(
            _coerce_dict(
                self._request_json("GET", f"/smr/artifacts/{artifact_id}"),
                label="get_artifact",
            )
        )

    def get_artifact_content(
        self,
        artifact_id: str,
        *,
        disposition: str = "inline",
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content(
                "GET",
                f"/smr/artifacts/{artifact_id}/content",
                params=build_query_params(disposition=disposition),
            ),
            label="get_artifact_content",
        )

    def download_artifact(
        self,
        artifact_id: str,
        output_path: str | os.PathLike[str],
        *,
        disposition: str = "attachment",
    ) -> dict[str, Any]:
        payload = self.get_artifact_content(artifact_id, disposition=disposition)
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload.get("encoding") == "base64":
            path.write_bytes(base64.b64decode(str(payload.get("content") or "")))
        else:
            path.write_text(str(payload.get("content") or ""), encoding="utf-8")
        return {
            "output_path": str(path),
            "artifact_id": artifact_id,
            "content_type": payload.get("content_type"),
            "bytes_written": path.stat().st_size,
        }

    def list_run_models(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/models"
            if project_id
            else f"/smr/runs/{run_id}/models"
        )
        return _coerce_dict_list(
            self._request_json("GET", path),
            label="list_run_models",
        )

    def list_run_datasets(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/datasets"
            if project_id
            else f"/smr/runs/{run_id}/datasets"
        )
        return _coerce_dict_list(
            self._request_json("GET", path),
            label="list_run_datasets",
        )

    def _legacy_list_project_run_artifacts(
        self,
        project_id: str,
        run_id: str,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/artifacts",
                params=build_query_params(
                    artifact_type=artifact_type,
                    limit=limit,
                    cursor=cursor,
                ),
            ),
            label="list_project_run_artifacts",
        )

    def _get_run_output_file_content(
        self,
        run_id: str,
        output_file_id: str,
        *,
        disposition: str = "inline",
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content(
                "GET",
                f"/smr/runs/{run_id}/outputs/{output_file_id}/content",
                params=build_query_params(disposition=disposition),
            ),
            label="get_run_output_file_content",
        )

    def list_project_external_repositories(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/external-repositories",
            ),
            label="list_project_external_repositories",
        )

    def list_project_repo_bindings(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/repos"),
            label="list_project_repo_bindings",
        )

    def attach_project_repo(
        self,
        project_id: str,
        *,
        repo: str,
        pr_write_enabled: bool = False,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/repos",
                json_body={
                    "repo": _require_non_empty_string(repo, field_name="repo"),
                    "pr_write_enabled": bool(pr_write_enabled),
                },
            ),
            label="attach_project_repo",
        )

    def detach_project_repo(self, project_id: str, *, repo: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "DELETE",
                f"/smr/projects/{project_id}/repos",
                json_body={"repo": _require_non_empty_string(repo, field_name="repo")},
            ),
            label="detach_project_repo",
        )

    def list_environments(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                "/smr/environments",
                params=build_query_params(limit=limit),
            ),
            label="list_environments",
        )

    def get_environment(self, *, name: str, digest: str | None = None) -> dict[str, Any]:
        name = _require_non_empty_string(name, field_name="name")
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/environments/{name}",
                params=build_query_params(digest=digest),
            ),
            label="get_environment",
        )

    def preflight_environment(
        self,
        *,
        name: str,
        digest: str | None = None,
    ) -> dict[str, Any]:
        name = _require_non_empty_string(name, field_name="name")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/environments/{name}/preflight",
                params=build_query_params(digest=digest),
            ),
            label="preflight_environment",
        )

    def list_project_outputs(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/outputs"),
            label="list_project_outputs",
        )

    def list_run_work_products(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/runs/{run_id}/work-products"),
            label="list_run_work_products",
        )

    def get_run_work_product(self, work_product_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/work-products/{work_product_id}"),
            label="get_run_work_product",
        )

    def export_run_work_product(
        self,
        work_product_id: str,
        *,
        destination: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/work-products/{work_product_id}/exports",
                json_body={
                    "destination": dict(destination),
                    "idempotency_key": idempotency_key,
                },
            ),
            label="export_run_work_product",
        )

    def export_trained_model(
        self,
        model_id: str,
        *,
        destination: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/trained_models/{model_id}/exports",
                json_body={
                    "destination": dict(destination),
                    "idempotency_key": idempotency_key,
                },
            ),
            label="export_trained_model",
        )

    def create_trained_model_adapter_upload_url(
        self,
        model_id: str,
        *,
        expires_in: int = 3600,
        content_type: str = "application/gzip",
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/trained_models/{model_id}/adapter_upload_url",
                json_body={
                    "expires_in": expires_in,
                    "content_type": content_type,
                },
            ),
            label="create_trained_model_adapter_upload_url",
        )

    def complete_trained_model_adapter_upload(
        self,
        model_id: str,
        *,
        bucket: str,
        key: str,
        adapter_size_bytes: int,
        metadata_patch: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "bucket": bucket,
            "key": key,
            "adapter_size_bytes": adapter_size_bytes,
        }
        if metadata_patch is not None:
            body["metadata_patch"] = dict(metadata_patch)
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/trained_models/{model_id}/adapter_uploads/complete",
                json_body=body,
            ),
            label="complete_trained_model_adapter_upload",
        )

    def get_work_product_export(self, export_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/work-product-exports/{export_id}"),
            label="get_work_product_export",
        )

    def explain_work_product_blocker(self, work_product_id: str) -> dict[str, Any]:
        product = self.get_run_work_product(work_product_id)
        return {
            "work_product_id": work_product_id,
            "status": product.get("status"),
            "readiness": product.get("readiness"),
            "blocker": product.get("blocker"),
            "next_action": (
                "Resolve the recorded blocker, then retry validation or export."
                if product.get("blocker")
                else "No blocker is currently recorded."
            ),
        }

    def upload_container_eval_package(
        self,
        project_id: str,
        run_id: str,
        *,
        kind: str,
        name: str,
        version: str | None = None,
        artifact_id: str | None = None,
        storage_uri: str | None = None,
        archive_size_bytes: int | None = None,
        manifest: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/container-eval-packages",
                json_body={
                    "kind": kind,
                    "name": name,
                    "version": version,
                    "artifact_id": artifact_id,
                    "storage_uri": storage_uri,
                    "archive_size_bytes": archive_size_bytes,
                    "manifest": dict(manifest or {}),
                    "metadata": dict(metadata or {}),
                },
            ),
            label="upload_container_eval_package",
        )

    def list_run_container_eval_packages(
        self,
        project_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/container-eval-packages",
            ),
            label="list_run_container_eval_packages",
        )

    def validate_container_eval_package(self, package_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/container-eval-packages/{package_id}/validate",
            ),
            label="validate_container_eval_package",
        )

    def list_project_prs(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/prs"),
            label="list_project_prs",
        )

    def get_project_pr(self, project_id: str, pr_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/prs/{pr_id}"),
            label="get_project_pr",
        )

    def list_project_models(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/models"),
            label="list_project_models",
        )

    def get_project_model(self, project_id: str, model_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/models/{model_id}"),
            label="get_project_model",
        )

    def download_project_model(self, project_id: str, model_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_content(
                "GET",
                f"/smr/projects/{project_id}/models/{model_id}/download",
            ),
            label="download_project_model",
        )

    def list_export_targets(self) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", "/smr/exports/targets"),
            label="list_export_targets",
        )

    def create_export_target(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", "/smr/exports/targets", json_body=dict(payload)),
            label="create_export_target",
        )

    def patch_export_target(
        self,
        target_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/exports/targets/{target_id}",
                json_body=dict(payload),
            ),
            label="patch_export_target",
        )

    def delete_export_target(self, target_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("DELETE", f"/smr/exports/targets/{target_id}"),
            label="delete_export_target",
        )

    def get_project_export_binding(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/exports/binding"),
            label="get_project_export_binding",
        )

    def put_project_export_binding(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PUT",
                f"/smr/projects/{project_id}/exports/binding",
                json_body=dict(payload),
            ),
            label="put_project_export_binding",
        )

    def export_project_model(self, project_id: str, model_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/models/{model_id}/export",
            ),
            label="export_project_model",
        )

    def create_project_external_repository(
        self,
        project_id: str,
        *,
        name: str,
        url: str,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": _require_non_empty_string(name, field_name="name"),
            "url": _require_non_empty_string(url, field_name="url"),
        }
        if default_branch and default_branch.strip():
            payload["default_branch"] = default_branch.strip()
        if role and role.strip():
            payload["role"] = role.strip()
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/external-repositories",
                json_body=payload,
            ),
            label="create_project_external_repository",
        )

    def patch_project_external_repository(
        self,
        project_id: str,
        repository_id: str,
        *,
        url: str | None = None,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if url is not None:
            payload["url"] = url
        if default_branch is not None:
            payload["default_branch"] = default_branch
        if role is not None:
            payload["role"] = role
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
                json_body=payload,
            ),
            label="patch_project_external_repository",
        )

    def delete_project_external_repository(
        self,
        project_id: str,
        repository_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "DELETE",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
            ),
            label="delete_project_external_repository",
        )

    def list_run_repository_mounts(self, run_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/repository-mounts"),
            label="list_run_repository_mounts",
        )

    def create_run_repository_mount(
        self,
        run_id: str,
        *,
        repository_id: str,
        mount_name: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "repository_id": _require_non_empty_string(
                repository_id,
                field_name="repository_id",
            )
        }
        if mount_name and mount_name.strip():
            payload["mount_name"] = mount_name.strip()
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/repository-mounts",
                json_body=payload,
            ),
            label="create_run_repository_mount",
        )

    def list_project_credential_refs(
        self,
        project_id: str,
        *,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/credential-refs",
                params=build_query_params(kind=kind),
            ),
            label="list_project_credential_refs",
        )

    def create_project_credential_ref(
        self,
        project_id: str,
        *,
        kind: str,
        label: str,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": _require_non_empty_string(kind, field_name="kind"),
            "label": _require_non_empty_string(label, field_name="label"),
        }
        if provider is not None:
            payload["provider"] = provider
        if funding_source is not None:
            payload["funding_source"] = funding_source
        if credential_name is not None:
            payload["credential_name"] = credential_name
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/credential-refs",
                json_body=payload,
            ),
            label="create_project_credential_ref",
        )

    def patch_project_credential_ref(
        self,
        project_id: str,
        credential_ref_id: str,
        *,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if provider is not None:
            payload["provider"] = provider
        if funding_source is not None:
            payload["funding_source"] = funding_source
        if credential_name is not None:
            payload["credential_name"] = credential_name
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/credential-refs/{credential_ref_id}",
                json_body=payload,
            ),
            label="patch_project_credential_ref",
        )

    def delete_project_credential_ref(
        self,
        project_id: str,
        credential_ref_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "DELETE",
                f"/smr/projects/{project_id}/credential-refs/{credential_ref_id}",
            ),
            label="delete_project_credential_ref",
        )

    def list_run_credential_bindings(self, run_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/credential-bindings"),
            label="list_run_credential_bindings",
        )

    def create_run_credential_binding(
        self,
        run_id: str,
        *,
        credential_ref_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/credential-bindings",
                json_body={
                    "credential_ref_id": _require_non_empty_string(
                        credential_ref_id,
                        field_name="credential_ref_id",
                    )
                },
            ),
            label="create_run_credential_binding",
        )

    def get_project_setup(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/setup"),
            label="get_project_setup",
        )

    def get_project_setup_authority(self, project_id: str) -> dict[str, Any]:
        return self.get_project_setup(project_id)

    def prepare_project_setup(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/setup/prepare"),
            label="prepare_project_setup",
        )

    def prepare_project_setup_authority(self, project_id: str) -> dict[str, Any]:
        return self.prepare_project_setup(project_id)

    def start_project_onboarding(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/onboarding/start",
            ),
            label="start_project_onboarding",
        )

    def complete_project_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/onboarding/complete_step",
                json_body={
                    "step": _require_non_empty_string(step, field_name="step"),
                    "status": _require_non_empty_string(status, field_name="status"),
                    "detail": dict(detail or {}),
                },
            ),
            label="complete_project_onboarding_step",
        )

    def run_project_onboarding_dry_run(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/onboarding/dry_run",
            ),
            label="run_project_onboarding_dry_run",
        )

    def get_project_onboarding_status(self, project_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/onboarding/status",
            ),
            label="get_project_onboarding_status",
        )

    def set_provider_key(
        self,
        project_id: str,
        *,
        provider: SmrCredentialProvider | str,
        funding_source: SmrFundingSource | str = SmrFundingSource.CUSTOMER_BYOK,
        api_key: str | None = None,
        encrypted_key_b64: str | None = None,
    ) -> dict[str, Any]:
        normalized_provider = _require_smr_credential_provider(
            provider,
            field_name="provider",
        )
        normalized_funding_source = _require_smr_funding_source(
            funding_source,
            field_name="funding_source",
        )
        payload: dict[str, Any] = {
            "provider": normalized_provider.value,
            "funding_source": normalized_funding_source.value,
        }
        if api_key and api_key.strip():
            payload["api_key"] = api_key.strip()
        if encrypted_key_b64 and encrypted_key_b64.strip():
            payload["encrypted_key_b64"] = encrypted_key_b64.strip()
        if "api_key" not in payload and "encrypted_key_b64" not in payload:
            raise ValueError("set_provider_key requires api_key or encrypted_key_b64")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/provider_keys",
                json_body=payload,
            ),
            label="set_provider_key",
        )

    def get_provider_key_status(
        self,
        project_id: str,
        *,
        provider: SmrCredentialProvider | str,
        funding_source: SmrFundingSource | str = SmrFundingSource.CUSTOMER_BYOK,
    ) -> dict[str, Any]:
        normalized_provider = _require_smr_credential_provider(
            provider,
            field_name="provider",
        )
        normalized_funding_source = _require_smr_funding_source(
            funding_source,
            field_name="funding_source",
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                (
                    f"/smr/projects/{project_id}/provider_keys/"
                    f"{normalized_provider.value}/{normalized_funding_source.value}/status"
                ),
            ),
            label="get_provider_key_status",
        )

    def register_local_github_credential(
        self,
        project_id: str,
        *,
        repo: str,
        access_token: str,
        pr_write_enabled: bool = True,
    ) -> dict[str, Any]:
        repo_value = _require_non_empty_string(repo, field_name="repo")
        token_value = _require_non_empty_string(access_token, field_name="access_token")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/integrations/github/local-dev-token/link",
                json_body={
                    "repo": repo_value,
                    "pr_write_enabled": bool(pr_write_enabled),
                    "access_token": token_value,
                },
            ),
            label="register_local_github_credential",
        )

    def register_local_github_repo_credential(
        self,
        *,
        repo: str,
        access_token: str,
        pr_write_enabled: bool = True,
    ) -> dict[str, Any]:
        repo_value = _require_non_empty_string(repo, field_name="repo")
        token_value = _require_non_empty_string(access_token, field_name="access_token")
        return _coerce_dict(
            self._request_json(
                "POST",
                "/smr/integrations/github/local-dev/register",
                json_body={
                    "repo": repo_value,
                    "pr_write_enabled": bool(pr_write_enabled),
                    "access_token": token_value,
                },
            ),
            label="register_local_github_repo_credential",
        )

    def get_github_status(self) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", "/smr/github/status"),
            label="get_github_status",
        )

    def start_github_oauth(
        self,
        *,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if redirect_uri is not None:
            payload["redirect_uri"] = redirect_uri
        return _coerce_dict(
            self._request_json("POST", "/smr/github/oauth/start", json_body=payload),
            label="start_github_oauth",
        )

    def list_github_repos(
        self,
        *,
        page: int | None = None,
        per_page: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                "/smr/github/repos",
                params=build_query_params(page=page, per_page=per_page),
            ),
            label="list_github_repos",
        )
        return _coerce_dict_list(payload.get("repos") or [], label="list_github_repos.repos")

    def disconnect_github(self) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("POST", "/smr/github/disconnect"),
            label="disconnect_github",
        )

    def get_local_publication_readiness(
        self,
        project_id: str,
        *,
        repo: str | None = None,
        pr_write_enabled: bool = True,
    ) -> LocalPublicationReadiness:
        params = build_query_params(
            repo=str(repo or "").strip() or None,
            pr_write_enabled=bool(pr_write_enabled),
        )
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/integrations/github/local-publication-readiness",
                params=params,
            ),
            label="get_local_publication_readiness",
        )
        try:
            return LocalPublicationReadiness.from_wire(payload)
        except ValueError as exc:
            raise SmrApiError(f"Invalid local publication readiness payload: {exc}") from exc

    def get_launch_preflight(
        self,
        project_id: str,
        *,
        host_kind: SmrHostKind | str | None = None,
        work_mode: SmrWorkMode | str | None = None,
        mode: SmrWorkMode | str | None = None,
        intended_horizon_hours: SmrIntendedHorizonHours | int | None = None,
        providers: Iterable[ProviderBinding | str | Mapping[str, Any] | dict[str, Any]]
        | None = None,
        limit: UsageLimit | Mapping[str, Any] | dict[str, Any] | None = None,
        worker_pool_id: str | None = None,
        runbook: SmrRunbookKind | str | None = None,
        runbook_preset: str | None = None,
        runbook_config_id: str | None = None,
        local_execution: Mapping[str, Any] | dict[str, Any] | None = None,
        execution_profile: LocalExecutionProfile | Mapping[str, Any] | dict[str, Any] | None = None,
        timebox_seconds: int | None = None,
        agent_profile: str | None = None,
        agent_model: SmrAgentModel | str | None = None,
        agent_harness: SmrAgentHarness | None = None,
        agent_kind: SmrAgentKind | None = None,
        agent_model_params: Mapping[str, Any] | dict[str, Any] | None = None,
        actor_model_overrides: Iterable[
            SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]
        ]
        | None = None,
        roles: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None = None,
        initial_runtime_messages: Iterable[Mapping[str, Any] | dict[str, Any]] | None = None,
        workflow: Mapping[str, Any] | dict[str, Any] | None = None,
        sandbox_override: Mapping[str, Any] | dict[str, Any] | None = None,
        environment: Mapping[str, Any] | dict[str, Any] | None = None,
        run_policy: SmrRunPolicy | Mapping[str, Any] | dict[str, Any] | None = None,
        kickoff_contract: KickoffContract | Mapping[str, Any] | dict[str, Any] | None = None,
        resource_bindings: RunResourceBindings | Mapping[str, Any] | dict[str, Any] | None = None,
        ai_cache: Mapping[str, Any] | dict[str, Any] | None = None,
        primary_objective_id: str | None = None,
        primary_objective_kind: str | None = None,
        primary_parent_ref: Mapping[str, Any] | dict[str, Any] | None = None,
        primary_parent: Mapping[str, Any] | dict[str, Any] | None = None,
        effort_id: str | None = None,
        idempotency_key_run_create: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = _build_project_run_payload(
            host_kind=host_kind,
            work_mode=work_mode,
            mode=mode,
            intended_horizon_hours=intended_horizon_hours,
            providers=providers,
            limit=limit,
            worker_pool_id=worker_pool_id,
            runbook=runbook,
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            local_execution=local_execution,
            execution_profile=execution_profile,
            timebox_seconds=timebox_seconds,
            agent_profile=agent_profile,
            agent_model=agent_model,
            agent_harness=agent_harness,
            agent_kind=agent_kind,
            agent_model_params=agent_model_params,
            actor_model_overrides=actor_model_overrides,
            roles=roles,
            initial_runtime_messages=initial_runtime_messages,
            workflow=workflow,
            sandbox_override=sandbox_override,
            environment=environment,
            run_policy=run_policy,
            kickoff_contract=kickoff_contract,
            resource_bindings=resource_bindings,
            ai_cache=ai_cache,
            primary_objective_id=primary_objective_id,
            primary_objective_kind=primary_objective_kind,
            primary_parent_ref=primary_parent_ref,
            primary_parent=primary_parent,
            effort_id=effort_id,
            idempotency_key_run_create=idempotency_key_run_create,
            idempotency_key=idempotency_key,
        )
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/launch-preflight",
                json_body=payload,
            ),
            label="get_launch_preflight",
        )

    def get_one_off_launch_preflight(self, **kwargs: Any) -> dict[str, Any]:
        payload = _build_project_run_payload(**kwargs)
        return _coerce_dict(
            self._request_json(
                "POST",
                "/smr/runs:one-off/launch-preflight",
                json_body=payload,
            ),
            label="get_one_off_launch_preflight",
        )

    def get_run_start_blockers(
        self,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Backward-compatible alias for launch preflight readiness checks."""

        return self.get_launch_preflight(project_id, **kwargs)

    def list_runbook_presets(self) -> tuple[SmrRunbookPreset, ...]:
        payload = _coerce_dict(
            self._request_json("GET", "/smr/runbook-presets"),
            label="list_runbook_presets",
        )
        raw_presets = payload.get("runbook_presets", [])
        if not isinstance(raw_presets, list):
            raise SmrApiError(
                "Invalid runbook preset catalog payload: runbook_presets must be a list"
            )
        presets: list[SmrRunbookPreset] = []
        for item in raw_presets:
            if not isinstance(item, Mapping):
                raise SmrApiError(
                    "Invalid runbook preset catalog payload: preset entries must be objects"
                )
            try:
                presets.append(SmrRunbookPreset.from_wire(item))
            except ValueError as exc:
                raise SmrApiError(f"Invalid runbook preset catalog payload: {exc}") from exc
        return tuple(presets)

    def trigger_run(
        self,
        project_id: str,
        *,
        host_kind: SmrHostKind | str | None = None,
        work_mode: SmrWorkMode | str | None = None,
        mode: SmrWorkMode | str | None = None,
        intended_horizon_hours: SmrIntendedHorizonHours | int | None = None,
        providers: Iterable[ProviderBinding | str | Mapping[str, Any] | dict[str, Any]]
        | None = None,
        limit: UsageLimit | Mapping[str, Any] | dict[str, Any] | None = None,
        worker_pool_id: str | None = None,
        runbook: SmrRunbookKind | str | None = None,
        runbook_preset: str | None = None,
        runbook_config_id: str | None = None,
        local_execution: Mapping[str, Any] | dict[str, Any] | None = None,
        execution_profile: LocalExecutionProfile | Mapping[str, Any] | dict[str, Any] | None = None,
        timebox_seconds: int | None = None,
        agent_profile: str | None = None,
        agent_model: SmrAgentModel | str | None = None,
        agent_harness: SmrAgentHarness | None = None,
        agent_kind: SmrAgentKind | None = None,
        agent_model_params: Mapping[str, Any] | dict[str, Any] | None = None,
        actor_model_overrides: Iterable[
            SmrActorModelAssignment | Mapping[str, Any] | dict[str, Any]
        ]
        | None = None,
        roles: SmrRoleBindings | Mapping[str, Any] | dict[str, Any] | None = None,
        initial_runtime_messages: Iterable[Mapping[str, Any] | dict[str, Any]] | None = None,
        workflow: Mapping[str, Any] | dict[str, Any] | None = None,
        sandbox_override: Mapping[str, Any] | dict[str, Any] | None = None,
        environment: Mapping[str, Any] | dict[str, Any] | None = None,
        run_policy: SmrRunPolicy | Mapping[str, Any] | dict[str, Any] | None = None,
        kickoff_contract: KickoffContract | Mapping[str, Any] | dict[str, Any] | None = None,
        resource_bindings: RunResourceBindings | Mapping[str, Any] | dict[str, Any] | None = None,
        ai_cache: Mapping[str, Any] | dict[str, Any] | None = None,
        primary_objective_id: str | None = None,
        primary_objective_kind: str | None = None,
        primary_parent_ref: Mapping[str, Any] | dict[str, Any] | None = None,
        primary_parent: Mapping[str, Any] | dict[str, Any] | None = None,
        effort_id: str | None = None,
        idempotency_key_run_create: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = _build_project_run_payload(
            host_kind=host_kind,
            work_mode=work_mode,
            mode=mode,
            intended_horizon_hours=intended_horizon_hours,
            providers=providers,
            limit=limit,
            worker_pool_id=worker_pool_id,
            runbook=runbook,
            runbook_preset=runbook_preset,
            runbook_config_id=runbook_config_id,
            local_execution=local_execution,
            execution_profile=execution_profile,
            timebox_seconds=timebox_seconds,
            agent_profile=agent_profile,
            agent_model=agent_model,
            agent_harness=agent_harness,
            agent_kind=agent_kind,
            agent_model_params=agent_model_params,
            actor_model_overrides=actor_model_overrides,
            roles=roles,
            initial_runtime_messages=initial_runtime_messages,
            workflow=workflow,
            sandbox_override=sandbox_override,
            environment=environment,
            run_policy=run_policy,
            kickoff_contract=kickoff_contract,
            resource_bindings=resource_bindings,
            ai_cache=ai_cache,
            primary_objective_id=primary_objective_id,
            primary_objective_kind=primary_objective_kind,
            primary_parent_ref=primary_parent_ref,
            primary_parent=primary_parent,
            effort_id=effort_id,
            idempotency_key_run_create=idempotency_key_run_create,
            idempotency_key=idempotency_key,
        )
        return _coerce_dict(
            self._request_json("POST", f"/smr/projects/{project_id}/trigger", json_body=payload),
            label="trigger_run",
        )

    def start_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        """Start a Managed Research run using the product launch vocabulary."""

        return self.trigger_run(project_id, **kwargs)

    def trigger_one_off_run(self, **kwargs: Any) -> dict[str, Any]:
        payload = _build_project_run_payload(**kwargs)
        return _coerce_dict(
            self._request_json("POST", "/smr/runs:one-off", json_body=payload),
            label="trigger_one_off_run",
        )

    def list_runs(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        public_state: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        if active_only:
            return self.list_active_runs(project_id)
        params = build_query_params(
            public_state=public_state,
            limit=limit,
            cursor=cursor,
        )
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/runs", params=params),
            label="list_runs",
        )

    def list_active_runs(self, project_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/runs/active"),
            label="list_active_runs",
        )

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        state: str | None = None,
        active_only: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                "/smr/jobs",
                params=build_query_params(
                    project_id=project_id,
                    state=state,
                    active_only=active_only,
                    limit=limit,
                ),
            ),
            label="list_jobs",
        )

    def get_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}",
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="get_project_run")
        return _coerce_dict(self._request_json("GET", f"/smr/runs/{run_id}"), label="get_run")

    def get_project_run(self, project_id: str, run_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/projects/{project_id}/runs/{run_id}"),
            label="get_project_run",
        )

    def get_run_public_state(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> ManagedResearchRun:
        return ManagedResearchRun.from_wire(self.get_run(run_id, project_id=project_id))

    def get_run_observability_snapshot(
        self,
        project_id: str,
        run_id: str,
        *,
        since_event_seq: int | None = None,
        latest_runtime_message_seq: int | None = None,
        latest_runtime_event_id: str | None = None,
        detail_level: str = "full",
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        params = build_query_params(
            since_event_seq=since_event_seq,
            latest_runtime_message_seq=latest_runtime_message_seq,
            latest_runtime_event_id=latest_runtime_event_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/poll-summary",
                params=params,
            ),
            label="get_run_observability_snapshot",
        )
        try:
            return RunObservabilitySnapshot.from_wire(payload)
        except ValueError as exc:
            raise SmrApiError(f"Invalid run observability snapshot payload: {exc}") from exc

    def get_run_observability_snapshot_control(
        self,
        project_id: str,
        run_id: str,
        *,
        since_event_seq: int | None = None,
        latest_runtime_message_seq: int | None = None,
        latest_runtime_event_id: str | None = None,
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> RunObservabilitySnapshot:
        """Fetch the lightweight control-plane projection for an in-flight run."""

        return self.get_run_observability_snapshot(
            project_id,
            run_id,
            since_event_seq=since_event_seq,
            latest_runtime_message_seq=latest_runtime_message_seq,
            latest_runtime_event_id=latest_runtime_event_id,
            detail_level="control",
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_run_observability_snapshot_full(
        self,
        project_id: str,
        run_id: str,
        *,
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        """Fetch the heavy evidence projection; use after terminal state."""

        return self.get_run_observability_snapshot(
            project_id,
            run_id,
            detail_level="full",
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_run_contract(
        self,
        project_id: str,
        run_id: str,
    ) -> ManagedResearchRunContract:
        return self.get_run_observability_snapshot(
            project_id,
            run_id,
            event_limit=1,
            actor_limit=1,
            task_limit=1,
            question_limit=1,
            timeline_limit=1,
            message_limit=1,
        ).run_contract

    def get_run_execution(
        self,
        project_id: str,
        run_id: str,
        *,
        view: str = "summary",
        event_limit: int = 100,
        actor_limit: int = 50,
        task_limit: int = 100,
        message_limit: int = 50,
        work_product_limit: int = 50,
    ) -> RunExecutionProjection:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/execution",
                params=build_query_params(
                    view=view,
                    event_limit=event_limit,
                    actor_limit=actor_limit,
                    task_limit=task_limit,
                    message_limit=message_limit,
                    work_product_limit=work_product_limit,
                ),
            ),
            label="get_run_execution",
        )
        try:
            return RunExecutionProjection.from_wire(payload)
        except ValueError as exc:
            raise SmrApiError(f"Invalid run execution projection payload: {exc}") from exc

    def list_run_task_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/task-events",
                params=params or None,
            ),
            label="list_run_task_events",
        )

    def list_run_objective_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/objective-events",
                params=params or None,
            ),
            label="list_run_objective_events",
        )

    def get_run_work_graph(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        params = {"limit": int(limit)} if limit is not None else None
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/work-graph",
                params=params,
            ),
            label="get_run_work_graph",
        )

    def poll_run_observability_snapshot(
        self,
        project_id: str,
        run_id: str,
        *,
        cursor: RunObservationCursor | Mapping[str, Any] | None = None,
        detail_level: str = "full",
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        resolved_cursor = (
            cursor
            if isinstance(cursor, RunObservationCursor)
            else RunObservationCursor.from_wire(cursor or {})
        )
        return self.get_run_observability_snapshot(
            project_id,
            run_id,
            since_event_seq=resolved_cursor.latest_event_seq,
            latest_runtime_message_seq=resolved_cursor.latest_runtime_message_seq,
            latest_runtime_event_id=resolved_cursor.latest_runtime_event_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_run_transcript(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/runtime/transcript",
                params=params,
            ),
            label="get_run_transcript",
        )

    def stream_run_events(
        self,
        run_id: str,
        *,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ):
        params = build_query_params(
            transcript_cursor=transcript_cursor,
            view=view,
        )
        for event in self._stream_sse(
            f"/smr/runs/{run_id}/runtime/stream",
            params=params,
            last_event_id=last_event_id,
            timeout=timeout,
        ):
            yield RunRuntimeStreamEvent.from_sse(event)

    def get_run_primary_parent(self, run_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json("GET", f"/smr/runs/{run_id}/primary-parent"),
            label="get_run_primary_parent",
        )

    def list_run_objective_scopes(self, run_id: str) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/objective-scopes"),
            label="list_run_objective_scopes",
        )

    def register_run_objective_scope(
        self,
        run_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PUT",
                f"/smr/runs/{run_id}/objective-scopes",
                json_body=dict(payload),
            ),
            label="register_run_objective_scope",
        )

    def list_run_primary_parent_milestones(
        self,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/primary-parent/milestones",
                params=build_query_params(limit=limit),
            ),
            label="list_run_primary_parent_milestones",
        )

    def list_project_milestones(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        parent_kind: str | None = None,
        parent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/milestones",
                params=build_query_params(
                    run_id=run_id,
                    parent_kind=parent_kind,
                    parent_id=parent_id,
                    limit=limit,
                ),
            ),
            label="list_project_milestones",
        )

    def create_project_milestone(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/milestones",
                json_body=dict(payload),
            ),
            label="create_project_milestone",
        )

    def get_project_milestone(
        self,
        project_id: str,
        milestone_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/milestones/{milestone_id}",
            ),
            label="get_project_milestone",
        )

    def patch_project_milestone(
        self,
        project_id: str,
        milestone_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/milestones/{milestone_id}",
                json_body=dict(payload),
            ),
            label="patch_project_milestone",
        )

    def transition_project_milestone(
        self,
        project_id: str,
        milestone_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/milestones/{milestone_id}/transition",
                json_body=dict(payload),
            ),
            label="transition_project_milestone",
        )

    def list_project_experiments(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiments",
                params=build_query_params(run_id=run_id, limit=limit),
            ),
            label="list_project_experiments",
        )

    def create_project_experiment(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/experiments",
                json_body=dict(payload),
            ),
            label="create_project_experiment",
        )

    def get_project_experiment(
        self,
        project_id: str,
        experiment_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiments/{experiment_id}",
            ),
            label="get_project_experiment",
        )

    def patch_project_experiment(
        self,
        project_id: str,
        experiment_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/experiments/{experiment_id}",
                json_body=dict(payload),
            ),
            label="patch_project_experiment",
        )

    def list_project_experiment_runs(
        self,
        project_id: str,
        experiment_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/runs",
                params=build_query_params(limit=limit),
            ),
            label="list_project_experiment_runs",
        )

    def link_project_experiment_run(
        self,
        project_id: str,
        experiment_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/runs",
                json_body=dict(payload),
            ),
            label="link_project_experiment_run",
        )

    def list_project_experiment_container_runs(
        self,
        project_id: str,
        experiment_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/container-runs",
                params=build_query_params(limit=limit),
            ),
            label="list_project_experiment_container_runs",
        )

    def attach_project_experiment_container_run(
        self,
        project_id: str,
        experiment_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/container-runs",
                json_body=dict(payload),
            ),
            label="attach_project_experiment_container_run",
        )

    def list_project_experiment_results(
        self,
        project_id: str,
        *,
        experiment_id: str | None = None,
        metric: str | None = None,
        taskset_id: str | None = None,
        taskset_seed: int | None = None,
        comparison_cohort_key: str | None = None,
        truth_status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiment-results",
                params=build_query_params(
                    experiment_id=experiment_id,
                    metric=metric,
                    taskset_id=taskset_id,
                    taskset_seed=taskset_seed,
                    comparison_cohort_key=comparison_cohort_key,
                    truth_status=truth_status,
                    limit=limit,
                ),
            ),
            label="list_project_experiment_results",
        )

    def list_project_experiment_results_for_experiment(
        self,
        project_id: str,
        experiment_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/results",
                params=build_query_params(limit=limit),
            ),
            label="list_project_experiment_results_for_experiment",
        )

    def attach_project_experiment_result(
        self,
        project_id: str,
        experiment_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/experiments/{experiment_id}/results",
                json_body=dict(payload),
            ),
            label="attach_project_experiment_result",
        )

    def rank_project_experiment_results(
        self,
        project_id: str,
        *,
        metric: str,
        taskset_id: str | None = None,
        taskset_seed: int | None = None,
        comparison_cohort_key: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/experiment-results/top",
                params=build_query_params(
                    metric=metric,
                    taskset_id=taskset_id,
                    taskset_seed=taskset_seed,
                    comparison_cohort_key=comparison_cohort_key,
                    limit=limit,
                ),
            ),
            label="rank_project_experiment_results",
        )

    def list_open_ended_questions(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/open-ended-questions",
                params=build_query_params(run_id=run_id, limit=limit),
            ),
            label="list_open_ended_questions",
        )

    def list_objectives(
        self,
        project_id: str,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives",
                params=build_query_params(kind=kind, run_id=run_id, limit=limit),
            ),
            label="list_objectives",
        )

    def create_objective(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives",
                json_body=dict(payload),
            ),
            label="create_objective",
        )

    def get_objective(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}",
                params=build_query_params(kind=kind),
            ),
            label="get_objective",
        )

    def get_objective_events(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/events",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="get_objective_events",
        )

    def get_objective_status(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        task_limit: int | None = None,
        claim_limit: int | None = None,
        event_limit: int | None = 50,
        milestone_limit: int | None = None,
    ) -> dict[str, Any]:
        objective = self.get_objective(project_id, objective_id, kind=kind)
        objective_kind = (
            str(kind or objective.get("objective_kind") or objective.get("kind") or "").strip()
            or None
        )
        progress = self.get_objective_progress(
            project_id,
            objective_id,
            kind=objective_kind,
        )
        tasks = self.list_objective_tasks(
            project_id,
            objective_id,
            kind=objective_kind,
            limit=task_limit,
        )
        claims = self.list_objective_claims(
            project_id,
            objective_id,
            kind=objective_kind,
            limit=claim_limit,
        )
        milestones = self.list_project_milestones(
            project_id,
            parent_kind=objective_kind,
            parent_id=objective_id,
            limit=milestone_limit,
        )
        events = self.get_objective_events(
            project_id,
            objective_id,
            kind=objective_kind,
            limit=event_limit,
        )
        latest_review = None
        for claim in claims:
            claim_kind = str(claim.get("claim_kind") or claim.get("kind") or "").strip()
            if "review" in claim_kind:
                latest_review = claim
                break
        blockers = [
            item
            for item in [*tasks, *milestones]
            if str(item.get("status") or item.get("state") or "").strip().lower()
            in {"blocked", "failed"}
        ]
        return {
            "objective": objective,
            "progress": progress,
            "related_tasks": tasks,
            "related_milestones": milestones,
            "claims": claims,
            "latest_review": latest_review,
            "blockers": blockers,
            "recent_events": events.get("events", []),
            "events": events,
            "run_scopes": objective.get("run_scopes") or objective.get("runs") or [],
        }

    def patch_objective(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/objectives/{objective_id}",
                params=build_query_params(kind=kind),
                json_body=dict(payload),
            ),
            label="patch_objective",
        )

    def pause_objective(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/pause",
                params=build_query_params(kind=kind),
            ),
            label="pause_objective",
        )

    def resume_objective(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/resume",
                params=build_query_params(kind=kind),
            ),
            label="resume_objective",
        )

    def withdraw_objective(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/withdraw",
                params=build_query_params(kind=kind),
            ),
            label="withdraw_objective",
        )

    def get_objective_progress(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/progress",
                params=build_query_params(kind=kind),
            ),
            label="get_objective_progress",
        )

    def list_objective_tasks(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/tasks",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="list_objective_tasks",
        )

    def list_tasks(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        objective_id: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if objective_id:
            return self.list_objective_tasks(
                project_id,
                objective_id,
                kind=kind,
                limit=limit,
            )
        if not run_id:
            raise ValueError("run_id or objective_id is required")
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/tasks",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="list_tasks",
        )

    def create_task(
        self,
        run_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.plan_tasks(tasks=[dict(payload)]),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def update_task(
        self,
        run_id: str,
        task_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        task_payload = {"task_id": task_id, **dict(payload)}
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.plan_tasks(tasks=[task_payload]),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def cancel_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        reason: str | None = None,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.set_task_state(
                task_id=task_id,
                state="stopped",
                reason=reason,
            ),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def reassign_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        assignee: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.update_task(
            run_id,
            task_id,
            {"assignee": assignee},
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def list_objective_claims(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/claims",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="list_objective_claims",
        )

    def create_objective_claim(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/claims",
                params=build_query_params(kind=kind),
                json_body=dict(payload),
            ),
            label="create_objective_claim",
        )

    def request_objective_review(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/request-review",
                params=build_query_params(kind=kind),
                json_body=dict(payload or {}),
            ),
            label="request_objective_review",
        )

    def create_open_ended_question(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/open-ended-questions",
                json_body=dict(payload),
            ),
            label="create_open_ended_question",
        )

    def get_open_ended_question(self, project_id: str, objective_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/open-ended-questions/{objective_id}",
            ),
            label="get_open_ended_question",
        )

    def patch_open_ended_question(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/open-ended-questions/{objective_id}",
                json_body=dict(payload),
            ),
            label="patch_open_ended_question",
        )

    def transition_open_ended_question(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/open-ended-questions/{objective_id}/transition",
                json_body=dict(payload),
            ),
            label="transition_open_ended_question",
        )

    def list_directed_effort_outcomes(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/directed-effort-outcomes",
                params=build_query_params(run_id=run_id, limit=limit),
            ),
            label="list_directed_effort_outcomes",
        )

    def create_directed_effort_outcome(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/directed-effort-outcomes",
                json_body=dict(payload),
            ),
            label="create_directed_effort_outcome",
        )

    def get_directed_effort_outcome(self, project_id: str, objective_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/directed-effort-outcomes/{objective_id}",
            ),
            label="get_directed_effort_outcome",
        )

    def patch_directed_effort_outcome(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/directed-effort-outcomes/{objective_id}",
                json_body=dict(payload),
            ),
            label="patch_directed_effort_outcome",
        )

    def transition_directed_effort_outcome(
        self,
        project_id: str,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/directed-effort-outcomes/{objective_id}/transition",
                json_body=dict(payload),
            ),
            label="transition_directed_effort_outcome",
        )

    def _run_lifecycle_control(
        self,
        *,
        method: str,
        path: str,
        label: str,
    ) -> dict[str, Any]:
        """POST a lifecycle control and translate 409 bodies to typed errors.

        The backend contract for pause/resume/stop returns HTTP 409 with a
        ``detail`` mapping of
        ``{error_code, message, retryable, current_state, run_id}`` when
        the transition is rejected. We surface that as
        :class:`ManagedResearchRunControlError` so callers can discriminate
        auth / config / transient failure modes without re-parsing strings.
        Any other error status is left to the transport's existing mapping.
        """

        try:
            return _coerce_dict(
                self._request_json(method, path),
                label=label,
            )
        except SmrApiError as exc:
            if exc.status_code != 409:
                raise
            response_text = exc.response_text
            if response_text is None or not response_text.strip():
                raise ValueError(
                    f"{label}: HTTP 409 but response body was empty; "
                    "expected detail mapping with error_code/message/retryable/current_state/run_id"
                ) from exc
            try:
                payload = _json.loads(response_text)
            except ValueError as parse_exc:
                raise ValueError(
                    f"{label}: HTTP 409 body was not valid JSON: {response_text!r}"
                ) from parse_exc
            raise ManagedResearchRunControlError.from_response(
                payload=payload,
                status_code=exc.status_code,
                response_text=response_text,
            ) from exc

    def stop_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/stop",
                label="stop_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/stop",
            label="stop_run",
        )

    def pause_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/pause",
                label="pause_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/pause",
            label="pause_run",
        )

    def resume_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/resume",
                label="resume_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/resume",
            label="resume_run",
        )

    def control_project_run_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "action": action,
            "reason": reason,
            "idempotency_key": idempotency_key,
        }
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_id}/control",
                json_body={key: value for key, value in body.items() if value is not None},
            ),
            label="control_project_run_actor",
        )

    def list_run_questions(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params = build_query_params(status_filter=status_filter, limit=limit, cursor=cursor)
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/questions",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict_list(scoped, label="list_project_run_questions")
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/questions", params=params),
            label="list_run_questions",
        )

    def respond_to_run_question(
        self,
        run_id: str,
        question_id: str,
        *,
        response_text: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "response_text": _require_non_empty_string(response_text, field_name="response_text")
        }
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/questions/{question_id}/respond",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="respond_project_run_question")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/questions/{question_id}/respond",
                json_body=payload,
            ),
            label="respond_run_question",
        )

    def list_run_approvals(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params = build_query_params(status_filter=status_filter, limit=limit, cursor=cursor)
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict_list(scoped, label="list_project_run_approvals")
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/approvals", params=params),
            label="list_run_approvals",
        )

    def approve_run_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(comment=comment) or {}
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/approve",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="approve_project_run_approval")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/approvals/{approval_id}/approve",
                json_body=payload,
            ),
            label="approve_run_approval",
        )

    def deny_run_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(comment=comment) or {}
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/deny",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="deny_project_run_approval")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/approvals/{approval_id}/deny",
                json_body=payload,
            ),
            label="deny_run_approval",
        )

    def _run_checkpoint_path(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        if project_id:
            base = f"/smr/projects/{project_id}/runs/{run_id}/checkpoints"
        else:
            base = f"/smr/runs/{run_id}/checkpoints"
        if checkpoint_id and checkpoint_id.strip():
            return f"{base}/{checkpoint_id.strip()}"
        return base

    def request_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(checkpoint_id=checkpoint_id, reason=reason)
        path = self._run_checkpoint_path(run_id, project_id=project_id)
        label = (
            "request_project_run_checkpoint" if project_id is not None else "request_run_checkpoint"
        )
        return _coerce_dict(
            self._request_json("POST", path, json_body=payload or {}),
            label=label,
        )

    def get_run_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        allow_not_found: bool = False,
    ) -> Checkpoint | None:
        path = self._run_checkpoint_path(
            run_id,
            project_id=project_id,
            checkpoint_id=checkpoint_id,
        )
        label = "get_project_run_checkpoint" if project_id is not None else "get_run_checkpoint"
        payload = self._request_json("GET", path, allow_not_found=allow_not_found)
        if payload is None:
            return None
        return Checkpoint.from_wire(_coerce_dict(payload, label=label))

    def wait_for_run_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> Checkpoint:
        checkpoint_id_text = _require_non_empty_string(
            checkpoint_id,
            field_name="checkpoint_id",
        )
        timeout = max(0.1, float(timeout_seconds))
        poll_interval = max(0.1, float(poll_interval_seconds))
        deadline = time.monotonic() + timeout
        last_state: str | None = None
        while True:
            checkpoint = self.get_run_checkpoint(
                run_id,
                checkpoint_id_text,
                project_id=project_id,
                allow_not_found=True,
            )
            if checkpoint is not None:
                state = str(checkpoint.state).strip().lower()
                if state in {"ready", "failed", "pruned"}:
                    return checkpoint
                last_state = checkpoint.state
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(poll_interval, deadline - now))
        last_state_suffix = f" (last_state={last_state})" if last_state else ""
        raise SmrApiError(
            f"Timed out waiting for checkpoint '{checkpoint_id_text}' to materialize{last_state_suffix}"
        )

    def create_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> Checkpoint:
        control_ack = self.request_run_checkpoint(
            run_id,
            project_id=project_id,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )
        resolved_checkpoint_id = _require_non_empty_string(
            str(control_ack.get("checkpoint_id") or checkpoint_id or ""),
            field_name="checkpoint_id",
        )
        return self.wait_for_run_checkpoint(
            run_id,
            resolved_checkpoint_id,
            project_id=project_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def list_run_checkpoints(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[Checkpoint]:
        path = self._run_checkpoint_path(run_id, project_id=project_id)
        label = "list_project_run_checkpoints" if project_id is not None else "list_run_checkpoints"
        return [
            Checkpoint.from_wire(item)
            for item in _coerce_dict_list(
                self._request_json("GET", path),
                label=label,
            )
        ]

    def restore_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        reason: str | None = None,
        mode: str = "in_place",
    ) -> dict[str, Any]:
        payload = build_query_params(
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            reason=reason,
            mode=mode,
        )
        if project_id:
            return _coerce_dict(
                self._request_json(
                    "POST",
                    f"/smr/projects/{project_id}/runs/{run_id}/restore",
                    json_body=payload or {},
                ),
                label="restore_project_run_checkpoint",
            )
        return _coerce_dict(
            self._request_json("POST", f"/smr/runs/{run_id}/restore", json_body=payload or {}),
            label="restore_run_checkpoint",
        )

    def get_run_logical_timeline(self, project_id: str, run_id: str) -> SmrLogicalTimeline:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/timeline",
            ),
            label="get_run_logical_timeline",
        )
        return SmrLogicalTimeline.from_wire(payload)

    def get_project_run_event_log(
        self,
        project_id: str,
        run_id: str,
        *,
        sources: list[str] | None = None,
        event_kinds: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> SmrRunEventLog:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/event-log",
                params=build_query_params(
                    sources=sources,
                    event_kinds=event_kinds,
                    statuses=statuses,
                    limit=limit,
                ),
            ),
            label="get_project_run_event_log",
        )
        return SmrRunEventLog.from_wire(payload)

    def get_run_authority_readouts(
        self,
        run_id: str,
        *,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/authority-readouts",
                params=build_query_params(include_runtime_authority=include_runtime_authority),
            ),
            label="get_run_authority_readouts",
        )
        return SmrAuthorityReadouts.from_wire(payload)

    def get_project_run_authority_readouts(
        self,
        project_id: str,
        run_id: str,
        *,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/authority-readouts",
                params=build_query_params(include_runtime_authority=include_runtime_authority),
            ),
            label="get_project_run_authority_readouts",
        )
        return SmrAuthorityReadouts.from_wire(payload)

    def get_project_run_operator_evidence(
        self,
        project_id: str,
        run_id: str,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/operator-evidence",
                params=build_query_params(
                    runtime_timeline_limit=runtime_timeline_limit,
                    logical_timeline_limit=logical_timeline_limit,
                    transcript_limit=transcript_limit,
                    reconciliation_limit=reconciliation_limit,
                ),
            ),
            label="get_project_run_operator_evidence",
        )

    def get_run_traces(self, run_id: str) -> SmrRunTraces:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/traces",
            ),
            label="get_run_traces",
        )
        return SmrRunTraces.from_wire(payload)

    def get_project_run_traces(self, project_id: str, run_id: str) -> SmrRunTraces:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/traces",
            ),
            label="get_project_run_traces",
        )
        return SmrRunTraces.from_wire(payload)

    def get_project_run_actor_trace(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
        *,
        cursor: str | None = None,
        live_cursor: str | None = None,
        limit: int | None = None,
        include_live: bool | None = None,
        include_traces: bool | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            live_cursor=live_cursor,
            limit=limit,
            include_live=include_live,
            include_traces=include_traces,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_key}/trace",
                params=params,
            ),
            label="get_project_run_actor_trace",
        )

    def get_project_run_actor_trace_index(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/trace-index",
            ),
            label="get_project_run_actor_trace_index",
        )

    def get_project_run_actors(
        self,
        project_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors",
            ),
            label="get_project_run_actors",
        )
        actors = payload.get("actors")
        if not isinstance(actors, list):
            raise SmrApiError("get_project_run_actors response missing actors list")
        return [dict(item) for item in actors if isinstance(item, Mapping)]

    def get_project_run_actor_raw_traces(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
    ) -> list[dict[str, Any]]:
        payload = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_key}/traces",
        )
        if not isinstance(payload, list):
            raise ValueError("get_project_run_actor_raw_traces expected a list response")
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def get_project_run_raw_trace_events(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        redaction_mode: str | None = None,
        reconstruct: bool | None = None,
        category: str | list[str] | None = None,
        method: str | list[str] | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            limit=limit,
            redaction_mode=redaction_mode,
            reconstruct=reconstruct,
            category=category,
            method=method,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/traces/{artifact_id}/events",
                params=params,
            ),
            label="get_project_run_raw_trace_events",
        )

    def create_project_run_raw_trace_download_url(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        *,
        expires_in: int | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/traces/{artifact_id}/download-url",
                params=build_query_params(expires_in=expires_in),
            ),
            label="create_project_run_raw_trace_download_url",
        )

    def download_project_run_raw_trace(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        destination: str | os.PathLike[str],
        *,
        expires_in: int | None = None,
    ) -> dict[str, Any]:
        url_payload = self.create_project_run_raw_trace_download_url(
            project_id,
            run_id,
            artifact_id,
            expires_in=expires_in,
        )
        url = str(url_payload.get("url") or "").strip()
        if not url:
            raise ValueError("download URL response did not include url")
        response = httpx.get(url, timeout=self.timeout_seconds, follow_redirects=True)
        if response.is_error:
            _raise_for_error_response(response)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(response.content)
        result = dict(url_payload)
        result["destination"] = str(destination_path)
        result["size_bytes"] = len(response.content)
        return result

    def get_run_actor_usage(self, run_id: str) -> SmrRunActorUsage:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/actors/usage",
            ),
            label="get_run_actor_usage",
        )
        return SmrRunActorUsage.from_wire(payload)

    def get_project_run_actor_usage(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrRunActorUsage:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/usage",
            ),
            label="get_project_run_actor_usage",
        )
        return SmrRunActorUsage.from_wire(payload)

    def list_run_participants(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunParticipants:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/participants"
            if project_id
            else f"/smr/runs/{run_id}/participants"
        )
        payload = _coerce_dict(
            self._request_json("GET", path),
            label="list_run_participants",
        )
        return SmrRunParticipants.from_wire(payload)

    def get_run_artifact_progress(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunArtifactProgress:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/artifact-progress"
            if project_id
            else f"/smr/runs/{run_id}/artifact-progress"
        )
        payload = _coerce_dict(
            self._request_json("GET", path),
            label="get_run_artifact_progress",
        )
        return SmrRunArtifactProgress.from_wire(payload)

    def list_run_actor_logs(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        actor_id: str | None = None,
        turn_id: str | None = None,
        kind: str | None = None,
        since: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SmrRunActorLogs:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/actor-logs"
            if project_id
            else f"/smr/runs/{run_id}/actor-logs"
        )
        params = build_query_params(
            actor_id=actor_id,
            turn_id=turn_id,
            kind=kind,
            since=since,
            cursor=cursor,
            limit=limit,
        )
        payload = _coerce_dict(
            self._request_json("GET", path, params=params),
            label="list_run_actor_logs",
        )
        return SmrRunActorLogs.from_wire(payload)

    def get_run_cost_summary(self, run_id: str) -> SmrRunCostSummary:
        payload = _coerce_dict(
            self.run_cost.summary(run_id),
            label="get_run_cost_summary",
        )
        return SmrRunCostSummary.from_wire(payload)

    def report_tinker_training_usage(
        self,
        run_id: str,
        *,
        actual_cost_usd: float | None = None,
        estimated_cost_usd: float | None = None,
        model: str | None = None,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        provider_result_id: str | None = None,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.run_cost.report_tinker_training_usage(
            run_id,
            actual_cost_usd=actual_cost_usd,
            estimated_cost_usd=estimated_cost_usd,
            model=model,
            task_id=task_id,
            idempotency_key=idempotency_key,
            provider_result_id=provider_result_id,
            request_id=request_id,
            metadata=metadata,
        )

    def branch_run_from_checkpoint(
        self,
        run_id: str | None = None,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        mode: SmrBranchMode | str = SmrBranchMode.EXACT,
        message: str | None = None,
        reason: str | None = None,
        title: str | None = None,
        source_node_id: str | None = None,
    ) -> SmrRunBranchResponse:
        request = _coerce_branch_request(
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            mode=mode,
            message=message,
            reason=reason,
            title=title,
            source_node_id=source_node_id,
        )
        if project_id is not None and run_id is None:
            raise ValueError("run_id is required when project_id is provided")
        if project_id and run_id:
            path = f"/smr/projects/{project_id}/runs/{run_id}/branches"
            label = "branch_project_run_from_checkpoint"
        elif run_id:
            path = f"/smr/runs/{run_id}/branches"
            label = "branch_run_from_checkpoint"
        else:
            path = "/smr/checkpoints/branches"
            label = "branch_checkpoint_reference"
        payload = _coerce_dict(
            self._request_json("POST", path, json_body=request.to_wire()),
            label=label,
        )
        return SmrRunBranchResponse.from_wire(payload)

    def list_runtime_messages(
        self,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if viewer_role and viewer_role.strip():
            params["viewer_role"] = viewer_role.strip()
        if limit is not None:
            params["limit"] = int(limit)
        if isinstance(viewer_target, str) and viewer_target.strip():
            params["viewer_target"] = [viewer_target.strip()]
        elif viewer_target is not None:
            cleaned_targets = [str(item).strip() for item in viewer_target if str(item).strip()]
            if cleaned_targets:
                params["viewer_target"] = cleaned_targets
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/runtime/messages",
                params=params or None,
            ),
            label="list_runtime_messages",
        )

    def _list_runtime_messages(
        self,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_runtime_messages(
            run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )

    def list_project_run_runtime_messages(
        self,
        project_id: str,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if viewer_role and viewer_role.strip():
            params["viewer_role"] = viewer_role.strip()
        if limit is not None:
            params["limit"] = int(limit)
        if isinstance(viewer_target, str) and viewer_target.strip():
            params["viewer_target"] = [viewer_target.strip()]
        elif viewer_target is not None:
            cleaned_targets = [str(item).strip() for item in viewer_target if str(item).strip()]
            if cleaned_targets:
                params["viewer_target"] = cleaned_targets
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/runtime/messages",
                params=params or None,
            ),
            label="list_project_run_runtime_messages",
        )

    def submit_runtime_intent(
        self,
        run_id: str,
        intent: RuntimeIntent | Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str | None = None,
        mode: str = "queue",
        body: str | None = None,
        causation_id: str | None = None,
    ) -> RuntimeIntentReceipt:
        if isinstance(intent, RuntimeIntent):
            intent_payload = intent.to_wire()
        elif isinstance(intent, Mapping):
            intent_payload = dict(intent)
        else:
            raise ValueError("intent must be a RuntimeIntent or mapping")
        json_body: dict[str, Any] = {
            "intent": intent_payload,
            "mode": str(mode or "queue").strip().lower(),
        }
        if body and body.strip():
            json_body["body"] = body.strip()
        if causation_id and causation_id.strip():
            json_body["causation_id"] = causation_id.strip()
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents"
        )
        return RuntimeIntentReceipt.from_wire(
            _coerce_dict(
                self._request_json("POST", path, json_body=json_body),
                label="submit_runtime_intent",
            )
        )

    def list_runtime_intents(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[RuntimeIntentView]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if limit is not None:
            params["limit"] = int(limit)
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents"
        )
        return [
            RuntimeIntentView.from_wire(item)
            for item in _coerce_dict_list(
                self._request_json("GET", path, params=params or None),
                label="list_runtime_intents",
            )
        ]

    def get_runtime_intent(
        self,
        run_id: str,
        runtime_intent_id: str,
        *,
        project_id: str | None = None,
    ) -> RuntimeIntentView:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents/{runtime_intent_id}"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents/{runtime_intent_id}"
        )
        return RuntimeIntentView.from_wire(
            _coerce_dict(
                self._request_json("GET", path),
                label="get_runtime_intent",
            )
        )

    def enqueue_runtime_message(
        self,
        run_id: str,
        *,
        topic: str | None = None,
        causation_id: str | None = None,
        mode: str | None = None,
        spawn_policy: str | None = None,
        sender: str | None = None,
        target: str | None = None,
        participant_session_id: str | None = None,
        action: str | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        for key, value in (
            ("topic", topic),
            ("causation_id", causation_id),
            ("mode", mode),
            ("spawn_policy", spawn_policy),
            ("sender", sender),
            ("target", target),
            ("participant_session_id", participant_session_id),
            ("action", action),
            ("body", body),
        ):
            if value and value.strip():
                json_body[key] = value.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/runtime/messages",
                json_body=json_body,
            ),
            label="enqueue_runtime_message",
        )

    def publish_manderqueue_message(
        self,
        run_id: str,
        *,
        project_id: str,
        intent: str = "queue",
        audience: Mapping[str, Any] | dict[str, Any] | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        message_kind: str = "runtime_message",
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        fallback_policy: str = "block",
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {
            "intent": str(intent or "queue").strip(),
            "audience": dict(audience or {"kind": "run"}),
            "message_kind": str(message_kind or "runtime_message").strip(),
            "fallback_policy": str(fallback_policy or "block").strip(),
        }
        for key, value in (
            ("body", body),
            ("thread_id", thread_id),
            ("parent_message_id", parent_message_id),
            ("idempotency_key", idempotency_key),
            ("correlation_id", correlation_id),
            ("causation_id", causation_id),
        ):
            if value and str(value).strip():
                json_body[key] = str(value).strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages",
                json_body=json_body,
            ),
            label="publish_manderqueue_message",
        )

    def send_message(
        self,
        run_id: str,
        *,
        project_id: str,
        intent: str = "queue",
        audience: Mapping[str, Any] | dict[str, Any] | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        message_kind: str = "runtime_message",
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        fallback_policy: str = "block",
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a product-level message queue message for a run."""

        return self.publish_manderqueue_message(
            run_id,
            project_id=project_id,
            intent=intent,
            audience=audience,
            body=body,
            payload=payload,
            message_kind=message_kind,
            thread_id=thread_id,
            parent_message_id=parent_message_id,
            fallback_policy=fallback_policy,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )

    def list_manderqueue_threads(
        self,
        run_id: str,
        *,
        project_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = {"limit": int(limit)} if limit is not None else None
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/threads",
                params=params,
            ),
            label="list_manderqueue_threads",
        )

    def list_manderqueue_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if thread_id and thread_id.strip():
            params["thread_id"] = thread_id.strip()
        if limit is not None:
            params["limit"] = int(limit)
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages",
                params=params or None,
            ),
            label="list_manderqueue_messages",
        )

    def list_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List product-level message queue messages for a run."""

        return self.list_manderqueue_messages(
            run_id,
            project_id=project_id,
            thread_id=thread_id,
            limit=limit,
        )

    def list_manderqueue_interactions(
        self,
        run_id: str,
        *,
        project_id: str,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if limit is not None:
            params["limit"] = int(limit)
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/interactions",
                params=params or None,
            ),
            label="list_manderqueue_interactions",
        )

    def respond_to_manderqueue_interaction(
        self,
        run_id: str,
        interaction_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        if body and body.strip():
            json_body["body"] = body.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/interactions/{interaction_id}/responses",
                json_body=json_body,
            ),
            label="respond_to_manderqueue_interaction",
        )

    def edit_manderqueue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        if body and body.strip():
            json_body["body"] = body.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages/{message_id}/edit",
                json_body=json_body,
            ),
            label="edit_manderqueue_message",
        )

    def edit_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Edit a product-level message queue message for a run."""

        return self.edit_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
            body=body,
            payload=payload,
        )

    def retract_manderqueue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages/{message_id}/retract",
                json_body={},
            ),
            label="retract_manderqueue_message",
        )

    def retract_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        """Retract a product-level message queue message for a run."""

        return self.retract_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
        )

    def _list_run_log_archives(
        self,
        project_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/runs/{run_id}/logs/archives"),
            label="list_run_log_archives",
        )


class SmrControlClient(SmrControlClientMixin, ManagedResearchClient):
    """Compatibility alias that retains the legacy synth-ai bridge surface.

    `ManagedResearchClient` is the canonical public name. `SmrControlClient`
    remains as a one-release alias and preserves the older default-backend +
    synth-ai bridge behavior for migration safety.
    """

    def __init__(
        self,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        openai_transport_mode: str = OPENAI_TRANSPORT_MODE_AUTO,
        openai_organization: str | None = None,
        openai_project: str | None = None,
        openai_request_id: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base,
            timeout_seconds=timeout_seconds,
        )
        self._initialize_openai_bridge(
            openai_transport_mode=openai_transport_mode,
            openai_organization=openai_organization,
            openai_project=openai_project,
            openai_request_id=openai_request_id,
        )

    def close(self) -> None:
        self.close_openai_bridge()
        super().close()


def first_id(items: Iterable[dict[str, Any]], key: str) -> str | None:
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
