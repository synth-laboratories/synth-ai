"""Public Research API for Synth.

Canonical entry: ``SynthClient().research``.
"""

from __future__ import annotations

from synth_ai.core.research.contracts import (
    ActorHarness as ResearchAgentHarness,
    ActorModel as ResearchAgentModel,
    EnvironmentKind as ResearchEnvironmentKind,
    ProjectId,
    ResearchProject,
    ResearchProjectCreateRequest,
    ResearchProjectPatchRequest,
    ResearchRunLaunchRequest,
    ResearchSwarm,
    ResearchSwarmBranchRequest,
    ResearchSwarmBranchResult,
    ResearchSwarmLaunchRequest,
    ResearchSwarmPreflight,
    ResearchSwarmState,
    ResourceLimit,
    RuntimeKind as ResearchRuntimeKind,
    SwarmId,
)
from synth_ai.core.research.contracts.swarms import HostKind as ResearchHostKind
from synth_ai.core.research.contracts.swarms import WorkMode as ResearchWorkMode
from synth_ai.core.research.projects import ResearchProjectsAPI
from synth_ai.core.research.swarms import ResearchSwarmHandle, ResearchSwarmsAPI
from synth_ai.research.async_client import AsyncResearchClient
from synth_ai.research.client import ResearchClient
from synth_ai.research.economics import ResearchEconomicsAPI
from synth_ai.research.errors import (
    ResearchApiError,
    ResearchConcurrentRunLimitExceededError,
    ResearchInsufficientCreditsError,
    ResearchLimitExceededError,
    ResearchProjectMonthlyBudgetExhaustedError,
    ResearchStructuredDenialError,
)
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.limits import ResearchLimitsAPI
from synth_ai.research.models import (
    ResearchAgentProfileBindings,
    ResearchArtifact,
    ResearchArtifactManifest,
    ResearchAuthorityReadouts,
    ResearchBillingCatalog,
    ResearchBillingDrawdown,
    ResearchBillingEntitlements,
    ResearchBillingPlan,
    ResearchOrgLimits,
    ResearchProjectEconomics,
    ResearchRoleBinding,
    ResearchRoleBindings,
    ResearchRun,
    ResearchRunbookPreset,
    ResearchRunProgress,
    ResearchTagSessionCreateRequest,
    ResearchWorkerRolePalette,
    ResearchWorkProduct,
)
from synth_ai.research.runs import (
    ResearchRunHandle,
    ResearchRunsAPI,
    ResearchRunSession,
)
from synth_ai.research.secrets import ResearchSecretsAPI

ResearchRunnableProjectRequest = ResearchProjectCreateRequest

__all__ = [
    "AsyncResearchClient",
    "ResearchArtifact",
    "ResearchArtifactManifest",
    "ResearchAgentHarness",
    "ResearchAgentModel",
    "ResearchAgentProfileBindings",
    "ResearchAuthorityReadouts",
    "ResearchBillingCatalog",
    "ResearchBillingDrawdown",
    "ResearchBillingEntitlements",
    "ResearchBillingPlan",
    "ResearchApiError",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchEconomicsAPI",
    "ResearchEnvironmentKind",
    "ResearchHostKind",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchOrgLimits",
    "ResearchProject",
    "ResearchProjectCreateRequest",
    "ResearchProjectPatchRequest",
    "ResearchProjectEconomics",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchFactoriesAPI",
    "ResearchLimitsAPI",
    "ResearchProjectsAPI",
    "ResearchRun",
    "ResearchRunLaunchRequest",
    "ResearchRunProgress",
    "ResearchRunHandle",
    "ResearchRunSession",
    "ResearchRunbookPreset",
    "ResearchRunnableProjectRequest",
    "ResearchSwarm",
    "ResearchSwarmBranchRequest",
    "ResearchSwarmBranchResult",
    "ResearchSwarmHandle",
    "ResearchSwarmLaunchRequest",
    "ResearchSwarmPreflight",
    "ResearchSwarmState",
    "ResearchSwarmsAPI",
    "ResearchRoleBinding",
    "ResearchRoleBindings",
    "ResearchRuntimeKind",
    "ResearchTagSessionCreateRequest",
    "ResearchRunsAPI",
    "ResearchSecretsAPI",
    "ResearchStructuredDenialError",
    "ResearchWorkMode",
    "ResearchWorkProduct",
    "ResearchWorkerRolePalette",
    "ProjectId",
    "ResourceLimit",
    "SwarmId",
]
