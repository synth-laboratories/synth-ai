"""Compatibility re-export; implementation lives in contracts.smr_agent_harnesses."""

from synth_ai.core.research.contracts.smr_agent_harnesses import (
    SMR_AGENT_HARNESS_VALUES,
    SmrAgentHarness,
    coerce_smr_agent_harness,
)

__all__ = [
    'SMR_AGENT_HARNESS_VALUES',
    'SmrAgentHarness',
    'coerce_smr_agent_harness',
]
