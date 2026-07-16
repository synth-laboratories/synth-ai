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
    ResearchTagSessionCreateRequest,
    ResearchWorkerRolePalette,
    ResearchWorkProduct,
)
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.runs import (
    ResearchRunHandle,
    ResearchRunsAPI,
    ResearchRunSession,
)
from synth_ai.research.secrets import ResearchSecretsAPI

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
    "ResearchWorkMode",
    "ResearchWorkProduct",
    "ResearchWorkerRolePalette",
]
