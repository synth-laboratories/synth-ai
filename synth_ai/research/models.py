"""Research API model aliases.

Typed wire models live in the canonical ``synth_ai.managed_research`` package.
"""

from __future__ import annotations

from synth_ai.managed_research.models.project import (
    CreateRunnableResult as ResearchCreateProjectResult,
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
