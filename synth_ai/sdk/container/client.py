"""Container client re-exports.

Prefer this module over synth_ai.sdk.container._impl.client.* moving forward.
"""

from __future__ import annotations

from synth_ai.sdk.container._impl.client import ContainerClient

__all__ = ["ContainerClient", "ContainerClient"]
