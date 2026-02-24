"""Canonical inference namespace.

# See: specs/sdk_logic.md
"""

from synth_ai.sdk.inference import (
    InferenceArtifactSpec,
    InferenceJobRequest,
)
from synth_ai.sdk.inference import (
    InferenceClient as Client,
)
from synth_ai.sdk.inference import (
    InferenceJobsClient as JobsClient,
)

__all__ = [
    "Client",
    "InferenceArtifactSpec",
    "InferenceJobRequest",
    "JobsClient",
]
