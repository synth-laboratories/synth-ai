"""Public Research API for Synth.

Canonical entry: ``SynthClient().research``.
"""

from __future__ import annotations

from synth_ai.research.async_client import AsyncResearchClient
from synth_ai.research.client import ResearchClient
from synth_ai.research.economics import ResearchEconomicsAPI
from synth_ai.research.enums import ResearchHostKind, ResearchWorkMode
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
    ResearchAgentHarness,
    ResearchAgentModel,
    ResearchAgentProfileBindings,
    ResearchArtifact,
    ResearchArtifactManifest,
    ResearchAuthorityReadouts,
    ResearchBillingCatalog,
    ResearchBillingDrawdown,
    ResearchBillingEntitlements,
    ResearchBillingPlan,
    ResearchCreateProjectResult,
    ResearchEnvironmentKind,
    ResearchOrgLimits,
    ResearchProject,
    ResearchProjectEconomics,
    ResearchRoleBinding,
    ResearchRoleBindings,
    ResearchRun,
    ResearchRunbookPreset,
    ResearchRunLaunchRequest,
    ResearchRunnableProjectRequest,
    ResearchRunProgress,
    ResearchRuntimeKind,
    ResearchSwarm,
    ResearchSwarmProgress,
    ResearchTagSessionCreateRequest,
    ResearchWorkerRolePalette,
    ResearchWorkProduct,
)
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.secrets import ResearchSecretsAPI
from synth_ai.research.swarms import (
    ResearchSwarmHandle,
    ResearchSwarmsAPI,
    ResearchSwarmSession,
    SwarmResult,
    SwarmRetryResult,
    swarm_state_is_terminal,
)

# Deprecated run-noun aliases (public noun is Swarm; import shims live in
# synth_ai.research.runs / run_readouts for old module paths).
ResearchRunHandle = ResearchSwarmHandle
ResearchRunSession = ResearchSwarmSession
ResearchRunsAPI = ResearchSwarmsAPI

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
    "ResearchCreateProjectResult",
    "ResearchEconomicsAPI",
    "ResearchEnvironmentKind",
    "ResearchHostKind",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchOrgLimits",
    "ResearchProject",
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
    "ResearchRoleBinding",
    "ResearchRoleBindings",
    "ResearchRuntimeKind",
    "ResearchTagSessionCreateRequest",
    "ResearchRunsAPI",
    "ResearchSecretsAPI",
    "ResearchStructuredDenialError",
    "ResearchSwarm",
    "ResearchSwarmHandle",
    "ResearchSwarmProgress",
    "ResearchSwarmSession",
    "ResearchSwarmsAPI",
    "ResearchWorkMode",
    "ResearchWorkProduct",
    "ResearchWorkerRolePalette",
    "SwarmResult",
    "SwarmRetryResult",
    "swarm_state_is_terminal",
]
