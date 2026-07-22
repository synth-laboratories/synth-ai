"""Concise public types for ``SynthClient().research``.

The hero workflow is ``Project -> Swarm -> events/result``.  Factory is the
durable optimization loop that creates and evaluates Efforts.  Names prefixed
with ``Research`` remain lazily importable through the 0.17 compatibility line
but are intentionally absent from public discovery.
"""

from __future__ import annotations

import importlib

from synth_ai.core.research.contracts import (
    ArtifactId,
    ActorTokenUsage,
    ActorUsage,
    ActorUsageMoney,
    ActorHarness,
    ActorModel,
    BudgetPolicy,
    CapacityPolicy,
    ContentDisposition,
    ConfigurationVersionId,
    Effort,
    EffortId,
    EffortPatch,
    EffortRecurrence,
    EffortSpec,
    EnvironmentKind,
    EvidenceArtifact,
    EvidenceFreshness,
    EvidenceWorkProduct,
    Factory,
    FactoryId,
    FactoryPatch,
    FactorySpec,
    FactoryTransition,
    HostKind,
    Project,
    ProjectId,
    ProjectPatch,
    ProjectSpec,
    ResolvedSwarmConfiguration,
    Swarm,
    SwarmEvidence,
    BranchSpec,
    BranchResult as Branch,
    SwarmSpec,
    SwarmPreflight as Preflight,
    SwarmState,
    SwarmUsage,
    TokenCounts,
    TokenUsage,
    UsageFreshness,
    UsageMoney,
    UsageSource,
    ResourceLimit as Limit,
    RuntimeKind,
    SwarmId,
    WorkMode,
    WorkProductArtifactLink,
    WorkProductArtifactRole,
    WorkProductBlocker,
    WorkProductId,
    WorkProductKind,
    WorkProductReadiness,
    WorkProductStatus,
)
from synth_ai.core.research.swarms import SwarmHandle
from synth_ai.core.errors import (
    RetryDirective as Retry,
    SynthError as Error,
    SynthErrorCategory as ErrorCategory,
    SynthErrorCode as ErrorCode,
    SynthFailure as Failure,
)
from synth_ai.core.research.events import (
    SwarmEvent,
    SwarmEventKind,
    SwarmTelemetry,
    UnknownSwarmEvent,
)
from synth_ai.research.async_client import AsyncClient
from synth_ai.research.client import Client

__all__ = [
    "ArtifactId",
    "ActorHarness",
    "ActorModel",
    "ActorTokenUsage",
    "ActorUsage",
    "ActorUsageMoney",
    "AsyncClient",
    "Branch",
    "BranchSpec",
    "BudgetPolicy",
    "CapacityPolicy",
    "Client",
    "ContentDisposition",
    "ConfigurationVersionId",
    "Effort",
    "EffortId",
    "EffortPatch",
    "EffortRecurrence",
    "EffortSpec",
    "EnvironmentKind",
    "EvidenceArtifact",
    "EvidenceFreshness",
    "EvidenceWorkProduct",
    "Error",
    "ErrorCategory",
    "ErrorCode",
    "Failure",
    "Factory",
    "FactoryId",
    "FactoryPatch",
    "FactorySpec",
    "FactoryTransition",
    "HostKind",
    "Limit",
    "Preflight",
    "Project",
    "ProjectId",
    "ProjectPatch",
    "ProjectSpec",
    "ResolvedSwarmConfiguration",
    "RuntimeKind",
    "Retry",
    "Swarm",
    "SwarmEvidence",
    "SwarmEvent",
    "SwarmEventKind",
    "SwarmHandle",
    "SwarmId",
    "SwarmSpec",
    "SwarmState",
    "SwarmTelemetry",
    "SwarmUsage",
    "TokenCounts",
    "TokenUsage",
    "UnknownSwarmEvent",
    "UsageFreshness",
    "UsageMoney",
    "UsageSource",
    "WorkMode",
    "WorkProductArtifactLink",
    "WorkProductArtifactRole",
    "WorkProductBlocker",
    "WorkProductId",
    "WorkProductKind",
    "WorkProductReadiness",
    "WorkProductStatus",
]

_COMPATIBILITY_EXPORTS: dict[str, tuple[str, str]] = {
    "AsyncResearchClient": ("synth_ai.research.async_client", "AsyncResearchClient"),
    "ResearchApiError": ("synth_ai.research.errors", "ResearchApiError"),
    "ResearchArtifact": ("synth_ai.research.models", "ResearchArtifact"),
    "ResearchArtifactManifest": ("synth_ai.research.models", "ResearchArtifactManifest"),
    "ResearchAgentHarness": ("synth_ai.core.research.contracts", "ActorHarness"),
    "ResearchAgentModel": ("synth_ai.core.research.contracts", "ActorModel"),
    "ResearchAgentProfileBindings": (
        "synth_ai.research.models",
        "ResearchAgentProfileBindings",
    ),
    "ResearchAuthorityReadouts": (
        "synth_ai.research.models",
        "ResearchAuthorityReadouts",
    ),
    "ResearchBillingCatalog": ("synth_ai.research.models", "ResearchBillingCatalog"),
    "ResearchBillingDrawdown": ("synth_ai.research.models", "ResearchBillingDrawdown"),
    "ResearchBillingEntitlements": (
        "synth_ai.research.models",
        "ResearchBillingEntitlements",
    ),
    "ResearchBillingPlan": ("synth_ai.research.models", "ResearchBillingPlan"),
    "ResearchClient": ("synth_ai.research.client", "ResearchClient"),
    "ResearchConcurrentRunLimitExceededError": (
        "synth_ai.research.errors",
        "ResearchConcurrentRunLimitExceededError",
    ),
    "ResearchEconomicsAPI": ("synth_ai.research.economics", "ResearchEconomicsAPI"),
    "ResearchEnvironmentKind": ("synth_ai.core.research.contracts", "EnvironmentKind"),
    "ResearchFactoriesAPI": ("synth_ai.research.factories", "ResearchFactoriesAPI"),
    "ResearchHostKind": ("synth_ai.core.research.contracts", "HostKind"),
    "ResearchInsufficientCreditsError": (
        "synth_ai.research.errors",
        "ResearchInsufficientCreditsError",
    ),
    "ResearchLimitExceededError": (
        "synth_ai.research.errors",
        "ResearchLimitExceededError",
    ),
    "ResearchLimitsAPI": ("synth_ai.research.limits", "ResearchLimitsAPI"),
    "ResearchOrgLimits": ("synth_ai.research.models", "ResearchOrgLimits"),
    "ResearchProject": ("synth_ai.core.research.contracts", "ResearchProject"),
    "ResearchProjectCreateRequest": (
        "synth_ai.core.research.contracts",
        "ResearchProjectCreateRequest",
    ),
    "ResearchProjectPatchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchProjectPatchRequest",
    ),
    "ResearchProjectEconomics": (
        "synth_ai.research.models",
        "ResearchProjectEconomics",
    ),
    "ResearchProjectMonthlyBudgetExhaustedError": (
        "synth_ai.research.errors",
        "ResearchProjectMonthlyBudgetExhaustedError",
    ),
    "ResearchProjectsAPI": ("synth_ai.core.research.projects", "ResearchProjectsAPI"),
    "ResearchRoleBinding": ("synth_ai.research.models", "ResearchRoleBinding"),
    "ResearchRoleBindings": ("synth_ai.research.models", "ResearchRoleBindings"),
    "ResearchRun": ("synth_ai.research.models", "ResearchRun"),
    "ResearchRunHandle": ("synth_ai.research.runs", "ResearchRunHandle"),
    "ResearchRunLaunchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchRunLaunchRequest",
    ),
    "ResearchRunProgress": ("synth_ai.research.models", "ResearchRunProgress"),
    "ResearchRunSession": ("synth_ai.research.runs", "ResearchRunSession"),
    "ResearchRunbookPreset": ("synth_ai.research.models", "ResearchRunbookPreset"),
    "ResearchRunnableProjectRequest": (
        "synth_ai.core.research.contracts",
        "ResearchProjectCreateRequest",
    ),
    "ResearchRunsAPI": ("synth_ai.research.runs", "ResearchRunsAPI"),
    "ResearchRuntimeKind": ("synth_ai.core.research.contracts", "RuntimeKind"),
    "ResearchSecretsAPI": ("synth_ai.research.secrets", "ResearchSecretsAPI"),
    "ResearchStructuredDenialError": (
        "synth_ai.research.errors",
        "ResearchStructuredDenialError",
    ),
    "ResearchSwarm": ("synth_ai.core.research.contracts", "ResearchSwarm"),
    "ResearchSwarmBranchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmBranchRequest",
    ),
    "ResearchSwarmBranchResult": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmBranchResult",
    ),
    "ResearchSwarmHandle": ("synth_ai.core.research.swarms", "ResearchSwarmHandle"),
    "ResearchSwarmLaunchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmLaunchRequest",
    ),
    "ResearchSwarmPreflight": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmPreflight",
    ),
    "ResearchSwarmState": ("synth_ai.core.research.contracts", "ResearchSwarmState"),
    "ResearchSwarmsAPI": ("synth_ai.core.research.swarms", "ResearchSwarmsAPI"),
    "ResearchTagSessionCreateRequest": (
        "synth_ai.research.models",
        "ResearchTagSessionCreateRequest",
    ),
    "ResearchWorkMode": ("synth_ai.core.research.contracts", "WorkMode"),
    "ResearchWorkerRolePalette": (
        "synth_ai.research.models",
        "ResearchWorkerRolePalette",
    ),
    "ResearchWorkProduct": ("synth_ai.research.models", "ResearchWorkProduct"),
    "ResourceLimit": ("synth_ai.core.research.contracts", "ResourceLimit"),
}


def __getattr__(name: str) -> object:
    target = _COMPATIBILITY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
