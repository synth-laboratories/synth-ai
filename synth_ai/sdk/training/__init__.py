"""Training SDK - prompt learning, SFT, RL, and ADAS jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA and MIPRO prompt optimization
- SFTJob: Supervised fine-tuning
- RLJob: Reinforcement learning (GSPO, GRPO, PPO, etc.)
- ADASJob: Automated Design of Agentic Systems (simplified workflows API)

Example:
    from synth_ai.sdk.training import PromptLearningJob, RLJob, ADASJob
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

    # ADAS workflow optimization (no task app needed)
    adas_job = ADASJob.from_dataset("my_tasks.json", rollout_budget=100)
    adas_job.submit()
    result = adas_job.stream_until_complete()
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

# ADAS
from synth_ai.sdk.api.train.adas import ADASJob, ADASJobResult, ADASSubmitResult
from synth_ai.sdk.api.train.adas_models import (
    ADASJobConfig,
    ADASTaskSet,
    ADASTask,
    ADASGoldOutput,
    ADASRubric,
    ADASJudgeConfig,
    load_adas_taskset,
    parse_adas_taskset,
)

__all__ = [
    # Prompt Learning
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    # RL
    "RLJob",
    "RLJobConfig",
    "RLJobPoller",
    # SFT
    "SFTJob",
    # ADAS
    "ADASJob",
    "ADASJobConfig",
    "ADASJobResult",
    "ADASSubmitResult",
    "ADASTaskSet",
    "ADASTask",
    "ADASGoldOutput",
    "ADASRubric",
    "ADASJudgeConfig",
    "load_adas_taskset",
    "parse_adas_taskset",
    # Utils
    "JobPoller",
    "PollOutcome",
]

