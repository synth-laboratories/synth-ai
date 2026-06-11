"""Research launch enums (public names)."""

from __future__ import annotations

from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode

ResearchWorkMode = SmrWorkMode
ResearchHostKind = SmrHostKind

SmrWorkMode = ResearchWorkMode
SmrHostKind = ResearchHostKind

__all__ = [
    "ResearchHostKind",
    "ResearchWorkMode",
    "SmrHostKind",
    "SmrWorkMode",
]
