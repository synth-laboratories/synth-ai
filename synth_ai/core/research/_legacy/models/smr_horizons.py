"""Compatibility re-export; implementation lives in contracts.smr_horizons."""

from synth_ai.core.research.contracts.smr_horizons import (
    SMR_INTENDED_HORIZON_HOURS_VALUES,
    SmrIntendedHorizonHours,
    coerce_intended_horizon_hours,
)

__all__ = [
    'SMR_INTENDED_HORIZON_HOURS_VALUES',
    'SmrIntendedHorizonHours',
    'coerce_intended_horizon_hours',
]
