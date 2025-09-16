from __future__ import annotations

"""
Compatibility layer: re-export Task App rollout contracts from synth_ai.task.contracts
so existing imports continue to work while consolidating under synth_ai.task.
"""

from synth_ai.task.contracts import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutSafetyConfig,
    RolloutRequest,
    RolloutStep,
    RolloutTrajectory,
    RolloutMetrics,
    RolloutResponse,
)

__all__ = [
    "RolloutEnvSpec",
    "RolloutPolicySpec",
    "RolloutRecordConfig",
    "RolloutSafetyConfig",
    "RolloutRequest",
    "RolloutStep",
    "RolloutTrajectory",
    "RolloutMetrics",
    "RolloutResponse",
]


