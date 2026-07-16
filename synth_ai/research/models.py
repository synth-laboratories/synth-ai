"""Research API model aliases.

Typed wire models live in the canonical ``synth_ai.managed_research.models`` package.
Import these names from ``synth_ai.research.models`` in customer code.

| Public name | Role |
| --- | --- |
| ``ResearchProject`` | Project record |
| ``ResearchRun`` | Run state model returned by ``runs.wait`` / ``runs.state`` |
| ``ResearchRunbookPreset`` | Named runbook preset for ``runs.create`` |
| ``ResearchOrgLimits`` | Organization plan limits and usage windows |
| ``ResearchBillingEntitlements`` | Organization billing entitlement snapshot |
| ``ResearchProjectEconomics`` | Project usage, entitlements, and budgets |
| ``ResearchRunProgress`` | Coarse typed progress returned by ``handle.progress.get`` |
| ``ResearchWorkProduct`` | Work product metadata |
| ``ResearchArtifact`` | Run artifact metadata |
| ``ResearchArtifactManifest`` | Typed run artifact manifest |
| ``ResearchTagSessionCreateRequest`` | Typed Factory Tag session request |
| ``ResearchCreateProjectResult`` | Result of runnable project creation |
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
from synth_ai.managed_research.models.run_state import ManagedResearchRun as ResearchRun
from synth_ai.managed_research.models.smr_runbooks import SmrRunbookPreset as ResearchRunbookPreset
from synth_ai.managed_research.models.tag import (
    TagSessionCreateRequest as ResearchTagSessionCreateRequest,
)
from synth_ai.managed_research.models.work_products import (
    ManagedResearchRunWorkProduct as ResearchWorkProduct,
)
from synth_ai.managed_research.models.types import RunArtifact as ResearchArtifact
from synth_ai.managed_research.models.types import (
    RunArtifactManifest as ResearchArtifactManifest,
)
from synth_ai.managed_research.models.types import RunProgress


@dataclass(frozen=True)
class ResearchRunProgress(RunProgress):
    """Public progress model including the backend-owned run state projection."""

    public_state: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ResearchRunProgress:
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
    "ResearchBillingEntitlements",
    "ResearchCreateProjectResult",
    "ResearchOrgLimits",
    "ResearchProject",
    "ResearchProjectEconomics",
    "ResearchRun",
    "ResearchRunProgress",
    "ResearchRunbookPreset",
    "ResearchTagSessionCreateRequest",
    "ResearchWorkProduct",
]
