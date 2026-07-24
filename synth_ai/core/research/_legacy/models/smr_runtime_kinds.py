"""Compatibility re-export; implementation lives in contracts.smr_runtime_kinds."""

from synth_ai.core.research.contracts.smr_runtime_kinds import (
    SMR_RUNTIME_KIND_VALUES,
    SmrRuntimeKind,
    coerce_smr_runtime_kind,
)

__all__ = [
    'SMR_RUNTIME_KIND_VALUES',
    'SmrRuntimeKind',
    'coerce_smr_runtime_kind',
]
