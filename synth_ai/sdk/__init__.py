"""Canonical SDK namespace.

# See: specifications/daily/feb24_2026/tinker_synth_final.md
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ContainerClient",
    "EvalJob",
    "GraphOptimizationJob",
    "GraphsClient",
    "InferenceClient",
    "InferenceJobsClient",
    "InProcessContainer",
    "OfflineJob",
    "OnlineSession",
    "PoolsClient",
    "System",
    "VerifiersClient",
    "create_container",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "System": ("synth_ai.optimization", "System"),
    "OfflineJob": ("synth_ai.optimization", "OfflineJob"),
    "OnlineSession": ("synth_ai.optimization", "OnlineSession"),
    "GraphOptimizationJob": ("synth_ai.sdk.optimization", "GraphOptimizationJob"),
    "EvalJob": ("synth_ai.sdk.eval", "EvalJob"),
    "InProcessContainer": ("synth_ai.container", "InProcessContainer"),
    "ContainerClient": ("synth_ai.container", "ContainerClient"),
    "create_container": ("synth_ai.container", "create_container"),
    "InferenceClient": ("synth_ai.inference", "Client"),
    "InferenceJobsClient": ("synth_ai.inference", "JobsClient"),
    "GraphsClient": ("synth_ai.graphs", "GraphsClient"),
    "VerifiersClient": ("synth_ai.verifiers", "VerifiersClient"),
    "PoolsClient": ("synth_ai.pools", "PoolsClient"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
