"""Training SDK - prompt learning, SFT, and RL jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA and MIPRO prompt optimization
- SFTJob: Supervised fine-tuning
- RLJob: Reinforcement learning (GRPO, PPO, etc.)

Example:
    from synth_ai.sdk.training import PromptLearningJob
    
    job = PromptLearningJob.from_config("config.toml")
    job.submit()
    result = job.poll_until_complete()
"""

from __future__ import annotations

# Re-export from existing locations
from synth_ai.sdk.api.train.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)
from synth_ai.sdk.api.train.sft import SFTJob

# Pollers and utilities
from synth_ai.sdk.api.train.pollers import JobPoller, PollOutcome

__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    "SFTJob",
    "JobPoller",
    "PollOutcome",
]

