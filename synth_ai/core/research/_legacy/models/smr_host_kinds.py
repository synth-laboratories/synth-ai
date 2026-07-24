"""Compatibility re-export; implementation lives in contracts.smr_host_kinds."""

from synth_ai.core.research.contracts.smr_host_kinds import (
    SMR_HOST_KIND_VALUES,
    SmrHostKind,
    coerce_smr_host_kind,
)

__all__ = [
    'SMR_HOST_KIND_VALUES',
    'SmrHostKind',
    'coerce_smr_host_kind',
]
