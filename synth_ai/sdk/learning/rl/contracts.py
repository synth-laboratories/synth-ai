"""Compatibility re-export for rollout contracts used by RL tooling."""

from __future__ import annotations

from synth_ai.sdk.task.contracts import (
    RolloutEnvSpec,
    RolloutMetrics,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutSafetyConfig,
    RolloutStep,
    RolloutTrajectory,
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
