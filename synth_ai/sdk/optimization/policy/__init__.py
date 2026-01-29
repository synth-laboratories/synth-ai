"""Policy optimization module (GEPA, MIPRO).

This module provides the canonical API for policy optimization jobs, which
optimize prompts/instructions using a LocalAPI for evaluation.

Algorithms:
- GEPA (Genetic Evolutionary Prompt Algorithm): Default algorithm
- MIPRO (Multi-step Instruction Optimization): Alternative algorithm

Example:
    >>> from synth_ai.sdk.optimization.policy import PolicyOptimizationJob
    >>>
    >>> job = PolicyOptimizationJob.from_config("config.toml")
    >>> job.submit()
    >>> result = job.stream_until_complete()
    >>> print(f"Best score: {result.best_score}")
"""

from __future__ import annotations

from .job import (
    PolicyOptimizationJob,
    PolicyOptimizationJobConfig,
    PolicyOptimizationResult,
)
from .mipro_online_session import MiproOnlineSession

__all__ = [
    "PolicyOptimizationJob",
    "PolicyOptimizationJobConfig",
    "PolicyOptimizationResult",
    "MiproOnlineSession",
]
