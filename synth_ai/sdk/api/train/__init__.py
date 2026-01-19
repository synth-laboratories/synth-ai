"""Training API for Prompt Learning (MIPRO/GEPA), GraphGen, and Context Learning.

This module provides both CLI and SDK interfaces for training jobs.

CLI Usage:
    uvx synth-ai train --type prompt_learning --config my_config.toml --poll

SDK Usage:
    from synth_ai.sdk.api.train import PromptLearningJob

    # Prompt Learning
    job = PromptLearningJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()

Note: SFT and RL training APIs have been moved to the research repo.
"""

# Re-export high-level SDK classes
from .context_learning import (
    ContextLearningJob,
    ContextLearningJobConfig,
)
from .graphgen import (
    GraphGenJob,
)
from .graphgen_models import (
    GraphGenJobConfig,
    GraphGenTaskSet,
)
from .prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)

__all__ = [
    # SDK - Prompt Learning
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    # SDK - Context Learning
    "ContextLearningJob",
    "ContextLearningJobConfig",
    # SDK - GraphGen
    "GraphGenJob",
    "GraphGenJobConfig",
    "GraphGenTaskSet",
]
