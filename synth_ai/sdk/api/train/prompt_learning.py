"""Backward-compatible PromptLearningJob import."""

from __future__ import annotations

from synth_ai.sdk.optimization.internal.prompt_learning import (  # noqa: F401
    PromptLearningJob,
    PromptLearningJobConfig,
)
from synth_ai.sdk.optimization.models import PromptLearningResult  # noqa: F401

__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningResult",
]
