"""Research launch enums (public names).

| Enum | Values | Use |
| --- | --- | --- |
| ``ResearchWorkMode`` | standard, … | ``runs.create(work_mode=...)`` |
| ``ResearchHostKind`` | container, … | Host selection on launch payloads |
"""

from __future__ import annotations

from synth_ai.core.research._legacy.models.smr_host_kinds import SmrHostKind
from synth_ai.core.research._legacy.models.smr_work_modes import SmrWorkMode

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
