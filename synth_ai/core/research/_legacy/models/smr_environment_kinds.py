"""Compatibility re-export; implementation lives in contracts.smr_environment_kinds."""

from synth_ai.core.research.contracts.smr_environment_kinds import (
    SMR_ENVIRONMENT_KIND_VALUES,
    SmrEnvironmentKind,
    coerce_smr_environment_kind,
)

__all__ = [
    'SMR_ENVIRONMENT_KIND_VALUES',
    'SmrEnvironmentKind',
    'coerce_smr_environment_kind',
]
