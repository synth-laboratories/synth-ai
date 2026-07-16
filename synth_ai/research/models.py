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
from synth_ai.managed_research.models.types import RunProgress as ResearchRunProgress

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
