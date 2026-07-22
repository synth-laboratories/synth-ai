"""Research API model aliases.

Typed wire models live in the canonical ``synth_ai.core.research._legacy.models`` package.
Import these names from ``synth_ai.core.research.models`` in customer code.

| Public name | Role |
| --- | --- |
| ``ResearchAgentHarness`` / ``ResearchAgentModel`` | Typed agent launch selectors |
| ``ResearchRoleBinding`` / ``ResearchRoleBindings`` / ``ResearchWorkerRolePalette`` | Typed role launch policy |
| ``ResearchRunLaunchRequest`` | Validated configured-run request |
| ``ResearchAuthorityReadouts`` | Canonical run/task authority projection |
| ``ResearchProject`` | Project record |
| ``ResearchRun`` | Run state model returned by ``runs.wait`` / ``runs.state`` |
| ``ResearchRunbookPreset`` | Named runbook preset for ``runs.create`` |
| ``ResearchOrgLimits`` | Organization plan limits and usage windows |
| ``ResearchBillingCatalog`` | Canonical billing plans and allowances catalog |
| ``ResearchBillingPlan`` | Organization billing plan snapshot |
| ``ResearchBillingDrawdown`` | Run or Factory-effort billing drawdown |
| ``ResearchBillingEntitlements`` | Organization billing entitlement snapshot |
| ``ResearchProjectEconomics`` | Project usage, entitlements, and budgets |
| ``ResearchRunProgress`` | Typed progress returned by ``handle.progress.get_typed`` |
| ``ResearchWorkProduct`` | Work product metadata |
| ``ResearchArtifact`` | Run artifact metadata |
| ``ResearchArtifactManifest`` | Typed run artifact manifest |
| ``ResearchTagSessionCreateRequest`` | Typed Factory Tag session request |
| ``ResearchCreateProjectResult`` | Result of runnable project creation |
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.core.research._legacy.models.project import (
    CreateRunnableResult as ResearchCreateProjectResult,
)
from synth_ai.core.research._legacy.models.project import (
    ManagedResearchProject as ResearchProject,
)
from synth_ai.core.research._legacy.models.run_launch import (
    RunLaunchRequest as ResearchRunLaunchRequest,
)
from synth_ai.core.research._legacy.models.run_state import ManagedResearchRun as ResearchRun
from synth_ai.core.research._legacy.models.run_timeline import (
    SmrAuthorityReadouts as ResearchAuthorityReadouts,
)
from synth_ai.core.research._legacy.models.smr_agent_harnesses import (
    SmrAgentHarness as ResearchAgentHarness,
)
from synth_ai.core.research._legacy.models.smr_agent_models import (
    SmrAgentModel as ResearchAgentModel,
)
from synth_ai.core.research._legacy.models.smr_environment_kinds import (
    SmrEnvironmentKind as ResearchEnvironmentKind,
)
from synth_ai.core.research._legacy.models.smr_roles import (
    RoleBinding as ResearchRoleBinding,
)
from synth_ai.core.research._legacy.models.smr_roles import (
    SmrRoleBindings as ResearchRoleBindings,
)
from synth_ai.core.research._legacy.models.smr_roles import (
    WorkerRolePalette as ResearchWorkerRolePalette,
)
from synth_ai.core.research._legacy.models.smr_runbooks import (
    SmrRunbookPreset as ResearchRunbookPreset,
)
from synth_ai.core.research._legacy.models.smr_runtime_kinds import (
    SmrRuntimeKind as ResearchRuntimeKind,
)
from synth_ai.core.research._legacy.models.tag import (
    TagSessionCreateRequest as ResearchTagSessionCreateRequest,
)
from synth_ai.core.research._legacy.models.types import RunArtifact as ResearchArtifact
from synth_ai.core.research._legacy.models.types import (
    RunArtifactManifest as ResearchArtifactManifest,
)
from synth_ai.core.research._legacy.models.types import (
    RunProgress,
)
from synth_ai.core.research._legacy.models.types import (
    SmrAgentProfileBindings as ResearchAgentProfileBindings,
)
from synth_ai.core.research._legacy.models.types import (
    SmrRunnableProjectRequest as ResearchRunnableProjectRequest,
)
from synth_ai.core.research._legacy.models.work_products import (
    ManagedResearchRunWorkProduct as ResearchWorkProduct,
)
from synth_ai.core.research.contracts.economics import (
    ResearchBillingCatalog,
    ResearchBillingDrawdown,
    ResearchBillingEntitlements,
    ResearchBillingPlan,
    ResearchOrgLimits,
    ResearchProjectEconomics,
)


@dataclass(frozen=True)
class ResearchRunProgress(RunProgress):
    """Public progress model including the backend-owned run state projection."""

    public_state: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ResearchRunProgress:
        """Decode backend run progress into the public progress model."""
        parsed = RunProgress.from_wire(payload)
        if not isinstance(payload, Mapping):
            raise ValueError("run progress must be an object")
        public_state = payload.get("public_state")
        if public_state is not None and not isinstance(public_state, str):
            raise ValueError("run progress public_state must be a string or null")
        return cls(**vars(parsed), public_state=public_state)


__all__ = [
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
    "ResearchCreateProjectResult",
    "ResearchEnvironmentKind",
    "ResearchOrgLimits",
    "ResearchProject",
    "ResearchProjectEconomics",
    "ResearchRun",
    "ResearchRunLaunchRequest",
    "ResearchRunProgress",
    "ResearchRunbookPreset",
    "ResearchRunnableProjectRequest",
    "ResearchRoleBinding",
    "ResearchRoleBindings",
    "ResearchRuntimeKind",
    "ResearchTagSessionCreateRequest",
    "ResearchWorkProduct",
    "ResearchWorkerRolePalette",
]
