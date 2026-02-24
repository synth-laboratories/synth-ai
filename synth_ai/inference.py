"""Canonical inference namespace.

# See: specifications/daily/feb24_2026/tinker_synth_final.md
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
