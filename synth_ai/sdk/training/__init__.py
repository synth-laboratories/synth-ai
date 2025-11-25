"""Training SDK - prompt learning, SFT, and RL jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA and MIPRO prompt optimization
- SFTJob: Supervised fine-tuning
- RLJob: Reinforcement learning (GSPO, GRPO, PPO, etc.)

Example:
    from synth_ai.sdk.training import PromptLearningJob, RLJob
    from synth_ai.sdk.task.in_process import InProcessTaskApp
    
    # Prompt optimization
    job = PromptLearningJob.from_config("config.toml")
    job.submit()
    result = job.poll_until_complete()
    
    # RL training with in-process task app
    async with InProcessTaskApp(task_app_path="my_task_app.py", port=8114) as task_app:
        rl_job = RLJob.from_config("rl_config.toml", task_app_url=task_app.url)
        rl_job.submit()
        rl_result = rl_job.poll_until_complete()
"""

from __future__ import annotations

# Pollers and utilities
from synth_ai.sdk.api.train.pollers import JobPoller, PollOutcome, RLJobPoller

# Re-export from existing locations
from synth_ai.sdk.api.train.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)
from synth_ai.sdk.api.train.rl import RLJob, RLJobConfig
from synth_ai.sdk.api.train.sft import SFTJob

__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    "RLJob",
    "RLJobConfig",
    "RLJobPoller",
    "SFTJob",
    "JobPoller",
    "PollOutcome",
]

