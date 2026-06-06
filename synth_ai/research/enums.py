"""Research launch enums (public names)."""

from __future__ import annotations

from managed_research.models.smr_host_kinds import SmrHostKind
from managed_research.models.smr_work_modes import SmrWorkMode

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
