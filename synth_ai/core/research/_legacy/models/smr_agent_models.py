"""Compatibility re-export; implementation lives in contracts.smr_agent_models."""

from synth_ai.core.research.contracts.smr_agent_models import (
    SMR_AGENT_MODEL_VALUES,
    SmrAgentModel,
    coerce_smr_agent_model,
)

__all__ = [
    'SMR_AGENT_MODEL_VALUES',
    'SmrAgentModel',
    'coerce_smr_agent_model',
]
