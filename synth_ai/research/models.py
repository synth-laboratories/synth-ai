"""Research API model aliases.

Typed wire models live in the canonical ``synth_ai.managed_research.models`` package.
Import these names from ``synth_ai.research.models`` in customer code.

| Public name | Role |
| --- | --- |
| ``ResearchProject`` | Project record |
| ``ResearchRun`` | Run state model returned by ``runs.wait`` / ``runs.state`` |
| ``ResearchRunbookPreset`` | Named runbook preset for ``runs.create`` |
| ``ResearchWorkProduct`` | Work product metadata |
| ``ResearchCreateProjectResult`` | Result of runnable project creation |
"""

from __future__ import annotations

from synth_ai.managed_research.models.project import (
    CreateRunnableResult as ResearchCreateProjectResult,
)
from synth_ai.managed_research.models.project import (
    ManagedResearchProject as ResearchProject,
)
from synth_ai.managed_research.models.run_state import ManagedResearchRun as ResearchRun
from synth_ai.managed_research.models.smr_runbooks import SmrRunbookPreset as ResearchRunbookPreset
from synth_ai.managed_research.models.work_products import (
    ManagedResearchRunWorkProduct as ResearchWorkProduct,
)

__all__ = [
    "ResearchCreateProjectResult",
    "ResearchProject",
    "ResearchRun",
    "ResearchRunbookPreset",
    "ResearchWorkProduct",
]
