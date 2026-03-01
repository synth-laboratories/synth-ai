"""Compatibility shim for Harbor provider imports.

Canonical module path:
- `synth_ai.sdk.container.harbor.provider`
"""

from .harbor.provider import HarborInstanceProvider, create_harbor_instance_provider

__all__ = [
    "HarborInstanceProvider",
    "create_harbor_instance_provider",
]
