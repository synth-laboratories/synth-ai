"""Compatibility re-export; implementation lives in contracts.smr_funding_sources."""

from synth_ai.core.research.contracts.smr_funding_sources import (
    SMR_FUNDING_SOURCE_VALUES,
    SmrFundingSource,
    coerce_smr_funding_source,
)

__all__ = [
    'SMR_FUNDING_SOURCE_VALUES',
    'SmrFundingSource',
    'coerce_smr_funding_source',
]
