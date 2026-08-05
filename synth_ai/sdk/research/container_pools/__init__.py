"""Container pools: deploy a container to a pool and run graded rollouts.

Reached as ``SynthClient().research.container_pools``. Note the neighbouring
``client.image_releases`` is a different object — that one is the SMR actor
runtime surface at ``/smr/v1/image-releases``; a pool's runtime image release
lives under ``container_pools.create_release``.
"""

from synth_ai.sdk.research.container_pools.api import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    ContainerPoolsAPI,
    RolloutTimeoutError,
)
from synth_ai.sdk.research.container_pools.contracts import (
    HARBOR_CONTAINER_SUBTYPE,
    TERMINAL_ROLLOUT_STATUSES,
    Pool,
    PoolId,
    PoolTask,
    PoolTaskId,
    Rollout,
    RolloutArtifact,
    RolloutId,
    RuntimeImageRelease,
    RuntimeImageReleaseId,
)
from synth_ai.sdk.research.container_pools.packaging import (
    HarborBundle,
    HarborBundleError,
    build_harbor_bundle_archive,
)
from synth_ai.sdk.research.contracts.managed_inference import (
    ManagedInference,
    ManagedInferenceLimits,
)

__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "HARBOR_CONTAINER_SUBTYPE",
    "TERMINAL_ROLLOUT_STATUSES",
    "ContainerPoolsAPI",
    "HarborBundle",
    "HarborBundleError",
    "Pool",
    "ManagedInference",
    "ManagedInferenceLimits",
    "PoolId",
    "PoolTask",
    "PoolTaskId",
    "Rollout",
    "RolloutArtifact",
    "RolloutId",
    "RolloutTimeoutError",
    "RuntimeImageRelease",
    "RuntimeImageReleaseId",
    "build_harbor_bundle_archive",
]
