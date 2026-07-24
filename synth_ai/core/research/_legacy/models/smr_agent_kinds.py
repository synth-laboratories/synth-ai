"""Compatibility re-export; implementation lives in contracts.smr_agent_kinds."""

from synth_ai.core.research.contracts.smr_agent_kinds import (
    SMR_AGENT_KIND_VALUES,
    SmrAgentKind,
    coerce_smr_agent_kind,
)

__all__ = [
    'SMR_AGENT_KIND_VALUES',
    'SmrAgentKind',
    'coerce_smr_agent_kind',
]
