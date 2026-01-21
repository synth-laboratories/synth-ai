"""Graph optimization module (Graph-GEPA).

This module provides the canonical API for graph optimization jobs, which
optimize multi-node workflow graphs using evolutionary algorithms.

Two construction modes:
- `from_config()`: Create from TOML config file
- `from_dataset()`: Create from JSON dataset (GraphEvolve style)

Example:
    >>> from synth_ai.sdk.optimization.graph import GraphOptimizationJob
    >>>
    >>> # From config file
    >>> job = GraphOptimizationJob.from_config("config.toml")
    >>> job.submit()
    >>> result = job.stream_until_complete()
    >>>
    >>> # From dataset
    >>> job = GraphOptimizationJob.from_dataset(
    ...     "tasks.json",
    ...     policy_models="gpt-4o-mini",
    ...     rollout_budget=100,
    ... )
    >>> job.submit()
    >>> result = job.stream_until_complete()
"""

from __future__ import annotations

from .job import (
    GraphOptimizationJob,
    GraphOptimizationJobConfig,
    GraphOptimizationResult,
)

__all__ = [
    "GraphOptimizationJob",
    "GraphOptimizationJobConfig",
    "GraphOptimizationResult",
]
