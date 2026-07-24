"""Compatibility re-export; implementation lives in contracts.smr_work_modes."""

from synth_ai.core.research.contracts.smr_work_modes import (
    SMR_WORK_MODE_VALUES,
    SmrWorkMode,
    coerce_smr_work_mode,
)

__all__ = [
    "SMR_WORK_MODE_VALUES",
    "SmrWorkMode",
    "coerce_smr_work_mode",
]
