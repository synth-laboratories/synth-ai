"""Research API model aliases.

Typed wire models live in the canonical ``synth_ai.managed_research.models`` package.
Import these names from ``synth_ai.research.models`` in customer code.

| Public name | Role |
| --- | --- |
| ``ResearchAgentHarness`` / ``ResearchAgentModel`` | Typed agent launch selectors |
| ``ResearchRoleBinding`` / ``ResearchRoleBindings`` / ``ResearchWorkerRolePalette`` | Typed role launch policy |
| ``ResearchRunLaunchRequest`` | Validated configured-run request |
| ``ResearchAuthorityReadouts`` | Canonical run/task authority projection |
| ``ResearchProject`` | Project record |
| ``ResearchSwarm`` | Swarm state model returned by ``swarms.wait`` / ``swarms.state`` (alias ``ResearchRun``) |
| ``ResearchRunbookPreset`` | Named runbook preset for ``swarms.create`` |
| ``ResearchOrgLimits`` | Organization plan limits and usage windows |
| ``ResearchBillingCatalog`` | Canonical billing plans and allowances catalog |
| ``ResearchBillingPlan`` | Organization billing plan snapshot |
| ``ResearchBillingDrawdown`` | Run or Factory-effort billing drawdown |
| ``ResearchBillingEntitlements`` | Organization billing entitlement snapshot |
| ``ResearchProjectEconomics`` | Project usage, entitlements, and budgets |
| ``ResearchSwarmProgress`` | Typed progress returned by ``handle.progress.get_typed`` (alias ``ResearchRunProgress``) |
| ``ResearchWorkProduct`` | Work product metadata |
| ``ResearchArtifact`` | Run artifact metadata |
| ``ResearchArtifactManifest`` | Typed run artifact manifest |
| ``ResearchTagSessionCreateRequest`` | Typed Factory Tag session request |
| ``ResearchCreateProjectResult`` | Result of runnable project creation |
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.managed_research.models.billing import (
    SmrBillingCatalog as ResearchBillingCatalog,
)
from synth_ai.managed_research.models.billing import (
    SmrBillingDrawdown as ResearchBillingDrawdown,
)
from synth_ai.managed_research.models.billing import (
    SmrBillingPlanSnapshot as ResearchBillingPlan,
)
from synth_ai.managed_research.models.canonical_usage import (
    BillingEntitlementSnapshot as ResearchBillingEntitlements,
)
from synth_ai.managed_research.models.canonical_usage import (
    OrgLimits as ResearchOrgLimits,
)
from synth_ai.managed_research.models.canonical_usage import (
    SmrProjectEconomics as ResearchProjectEconomics,
)
from synth_ai.managed_research.models.project import (
    CreateRunnableResult as ResearchCreateProjectResult,
)
from synth_ai.managed_research.models.project import (
    ManagedResearchProject as ResearchProject,
)
from synth_ai.managed_research.models.run_launch import (
    RunLaunchRequest as ResearchRunLaunchRequest,
)
from synth_ai.managed_research.models.run_state import ManagedResearchRun as ResearchSwarm
from synth_ai.managed_research.models.run_timeline import (
    SmrAuthorityReadouts as ResearchAuthorityReadouts,
)
from synth_ai.managed_research.models.smr_agent_harnesses import (
    SmrAgentHarness as ResearchAgentHarness,
)
from synth_ai.managed_research.models.smr_agent_models import (
    SmrAgentModel as ResearchAgentModel,
)
from synth_ai.managed_research.models.smr_environment_kinds import (
    SmrEnvironmentKind as ResearchEnvironmentKind,
)
from synth_ai.managed_research.models.smr_roles import (
    RoleBinding as ResearchRoleBinding,
)
from synth_ai.managed_research.models.smr_roles import (
    SmrRoleBindings as ResearchRoleBindings,
)
from synth_ai.managed_research.models.smr_roles import (
    WorkerRolePalette as ResearchWorkerRolePalette,
)
from synth_ai.managed_research.models.smr_runbooks import (
    SmrRunbookPreset as ResearchRunbookPreset,
)
from synth_ai.managed_research.models.smr_runtime_kinds import (
    SmrRuntimeKind as ResearchRuntimeKind,
)
from synth_ai.managed_research.models.tag import (
    TagSessionCreateRequest as ResearchTagSessionCreateRequest,
)
from synth_ai.managed_research.models.types import RunArtifact as ResearchArtifact
from synth_ai.managed_research.models.types import (
    RunArtifactManifest as ResearchArtifactManifest,
)
from synth_ai.managed_research.models.types import (
    RunProgress,
)
from synth_ai.managed_research.models.types import (
    SmrAgentProfileBindings as ResearchAgentProfileBindings,
)
from synth_ai.managed_research.models.types import (
    SmrRunnableProjectRequest as ResearchRunnableProjectRequest,
)
from synth_ai.managed_research.models.work_products import (
    ManagedResearchRunWorkProduct as ResearchWorkProduct,
)


@dataclass(frozen=True)
class ResearchSwarmProgress(RunProgress):
    """Public progress model including the backend-owned swarm state projection."""

    public_state: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ResearchSwarmProgress:
        """Decode backend swarm progress into the public progress model."""
        parsed = RunProgress.from_wire(payload)
        if not isinstance(payload, Mapping):
            raise ValueError("swarm progress must be an object")
        public_state = payload.get("public_state")
        if public_state is not None and not isinstance(public_state, str):
            raise ValueError("swarm progress public_state must be a string or null")
        return cls(**vars(parsed), public_state=public_state)


# Deprecated run-noun aliases (wire protocol still says "run"; public noun is Swarm).
ResearchRun = ResearchSwarm
ResearchRunProgress = ResearchSwarmProgress


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
    "ResearchSwarm",
    "ResearchSwarmProgress",
    "ResearchRunnableProjectRequest",
    "ResearchRoleBinding",
    "ResearchRoleBindings",
    "ResearchRuntimeKind",
    "ResearchTagSessionCreateRequest",
    "ResearchWorkProduct",
    "ResearchWorkerRolePalette",
]
