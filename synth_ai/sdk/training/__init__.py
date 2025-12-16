"""Training SDK - prompt learning, SFT, RL, and GraphGen jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA and MIPRO prompt optimization
- SFTJob: Supervised fine-tuning
- RLJob: Reinforcement learning (GSPO, GRPO, PPO, etc.)
- GraphGenJob: Automated Design of Agentic Systems (simplified workflows API)

Example:
    from synth_ai.sdk.training import PromptLearningJob, RLJob, GraphGenJob
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

    # GraphGen workflow optimization (no task app needed)
    graphgen_job = GraphGenJob.from_dataset("my_tasks.json", rollout_budget=100)
    graphgen_job.submit()
    result = graphgen_job.stream_until_complete()
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

# GraphGen (formerly GraphGen)
from synth_ai.sdk.api.train.graphgen import GraphGenJob, GraphGenJobResult, GraphGenSubmitResult
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenJobConfig,
    GraphGenTaskSet,
    GraphGenTask,
    GraphGenGoldOutput,
    GraphGenRubric,
    GraphGenJudgeConfig,
    load_graphgen_taskset,
    parse_graphgen_taskset,
    # GraphGen aliases
    GraphGenJobConfig,
    GraphGenTaskSet,
    GraphGenTask,
    GraphGenGoldOutput,
    GraphGenRubric,
    GraphGenJudgeConfig,
    load_graphgen_taskset,
    parse_graphgen_taskset,
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
    # GraphGen (preferred names)
    "GraphGenJobConfig",
    "GraphGenTaskSet",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenJudgeConfig",
    "load_graphgen_taskset",
    "parse_graphgen_taskset",
    # GraphGen (legacy aliases)
    "GraphGenJob",
    "GraphGenJobConfig",
    "GraphGenJobResult",
    "GraphGenSubmitResult",
    "GraphGenTaskSet",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenJudgeConfig",
    "load_graphgen_taskset",
    "parse_graphgen_taskset",
    # Utils
    "JobPoller",
    "PollOutcome",
]

