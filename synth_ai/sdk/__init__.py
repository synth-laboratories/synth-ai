"""Canonical SDK namespace.

# See: specs/sdk_logic.md
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ContainerClient",
    "InProcessContainer",
    "OfflineJob",
    "OnlineSession",
    "PoolsClient",
    "System",
    "create_container",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "System": ("synth_ai.optimization", "System"),
    "OfflineJob": ("synth_ai.optimization", "OfflineJob"),
    "OnlineSession": ("synth_ai.optimization", "OnlineSession"),
    "InProcessContainer": ("synth_ai.container", "InProcessContainer"),
    "ContainerClient": ("synth_ai.container", "ContainerClient"),
    "create_container": ("synth_ai.container", "create_container"),
    "PoolsClient": ("synth_ai.pools", "PoolsClient"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
