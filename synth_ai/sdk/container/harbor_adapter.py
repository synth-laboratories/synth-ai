"""Compatibility shim for Harbor adapter imports.

Canonical module path:
- `synth_ai.sdk.container.harbor.adapter`
"""

from .harbor.adapter import (
    HarborExecutionBackend,
    HarborExecutionError,
    create_harbor_rollout_executor,
)

__all__ = [
    "HarborExecutionBackend",
    "HarborExecutionError",
    "create_harbor_rollout_executor",
]
