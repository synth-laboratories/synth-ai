"""Research API model aliases (alpha bootstrap).

Typed wire models remain in ``managed_research`` until push 4 ports codecs here.
"""

from __future__ import annotations

from managed_research.models.project import (
    CreateRunnableResult as ResearchCreateProjectResult,
    ManagedResearchProject as ResearchProject,
)
from managed_research.models.run_state import ManagedResearchRun as ResearchRun
from managed_research.models.smr_runbooks import SmrRunbookPreset as ResearchRunbookPreset
from managed_research.models.work_products import (
    ManagedResearchRunWorkProduct as ResearchWorkProduct,
)

__all__ = [
    "ResearchCreateProjectResult",
    "ResearchProject",
    "ResearchRun",
    "ResearchRunbookPreset",
    "ResearchWorkProduct",
]
