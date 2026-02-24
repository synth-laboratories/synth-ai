"""Canonical pools namespace.

# See: specifications/daily/feb24_2026/tinker_synth_final.md
"""

from synth_ai.client import AsyncPoolsClient
from synth_ai.sdk.container_pools import ContainerPoolsClient as PoolsClient

__all__ = [
    "AsyncPoolsClient",
    "PoolsClient",
]
