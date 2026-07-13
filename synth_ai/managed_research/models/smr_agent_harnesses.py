"""Public agent-harness compatibility surface."""

from synth_ai.managed_research.models.smr_agent_kinds import (
    SMR_AGENT_KIND_VALUES as SMR_AGENT_HARNESS_VALUES,
)
from synth_ai.managed_research.models.smr_agent_kinds import (
    SmrAgentKind as SmrAgentHarness,
)
from synth_ai.managed_research.models.smr_agent_kinds import (
    coerce_smr_agent_kind as coerce_smr_agent_harness,
)

__all__ = [
    "SMR_AGENT_HARNESS_VALUES",
    "SmrAgentHarness",
    "coerce_smr_agent_harness",
]
