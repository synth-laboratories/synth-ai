"""Canonical SDK namespace.

# See: specs/sdk_logic.md
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AsyncContainerPoolsClient",
    "CANONICAL_ROLLOUT_REQUEST_KEYS",
    "ContainerClient",
    "ContainerPoolsClient",
    "InProcessContainer",
    "OfflineJob",
    "OnlineSession",
    "System",
    "create_container",
    "validate_pool_rollout_request",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "System": ("synth_ai.optimization", "System"),
    "OfflineJob": ("synth_ai.optimization", "OfflineJob"),
    "OnlineSession": ("synth_ai.optimization", "OnlineSession"),
    "AsyncContainerPoolsClient": ("synth_ai.sdk.container_pools", "AsyncContainerPoolsClient"),
    "CANONICAL_ROLLOUT_REQUEST_KEYS": ("synth_ai.sdk.container_pools", "CANONICAL_ROLLOUT_REQUEST_KEYS"),
    "InProcessContainer": ("synth_ai.container", "InProcessContainer"),
    "ContainerClient": ("synth_ai.container", "ContainerClient"),
    "ContainerPoolsClient": ("synth_ai.sdk.container_pools", "ContainerPoolsClient"),
    "create_container": ("synth_ai.container", "create_container"),
    "validate_pool_rollout_request": ("synth_ai.sdk.container_pools", "validate_pool_rollout_request"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
