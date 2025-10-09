from __future__ import annotations

"""Compatibility re-export for rollout contracts used by RL tooling."""

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

