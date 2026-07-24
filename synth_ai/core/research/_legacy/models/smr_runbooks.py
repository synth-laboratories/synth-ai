"""Compatibility re-export; implementation lives in contracts.smr_runbooks."""

from synth_ai.core.research.contracts.smr_runbooks import (
    SMR_RUNBOOK_KIND_VALUES,
    SmrRunbookKind,
    SmrRunbookLimitSummary,
    SmrRunbookPreset,
    coerce_smr_runbook_kind,
)

__all__ = [
    'SMR_RUNBOOK_KIND_VALUES',
    'SmrRunbookKind',
    'SmrRunbookLimitSummary',
    'SmrRunbookPreset',
    'coerce_smr_runbook_kind',
]
