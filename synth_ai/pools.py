"""Canonical container-pools namespace.

# See: specs/sdk_logic.md
"""

from synth_ai.sdk.container_pools import AsyncContainerPoolsClient, ContainerPoolsClient, PoolTarget

# Back-compat aliases; prefer ContainerPoolsClient/AsyncContainerPoolsClient.
PoolsClient = ContainerPoolsClient
AsyncPoolsClient = AsyncContainerPoolsClient

__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncPoolsClient",
    "ContainerPoolsClient",
    "PoolTarget",
    "PoolsClient",
]
