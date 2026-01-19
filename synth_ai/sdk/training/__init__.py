"""Training SDK - prompt learning and GraphGen jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA prompt optimization
- GraphGenJob: Graph Opt (simplified workflows API)

Note: SFT and RL training APIs have been moved to the research repo.

Example:
    from synth_ai.sdk.training import PromptLearningJob, GraphGenJob

    # Prompt optimization
    job = PromptLearningJob.from_config("config.toml")
    job.submit()
    result = job.poll_until_complete()

    # GraphGen workflow optimization (no task app needed)
    graphgen_job = GraphGenJob.from_dataset("my_tasks.json", rollout_budget=100)
    graphgen_job.submit()
    result = graphgen_job.stream_until_complete()
"""

from __future__ import annotations

# GraphGen (Graph Opt)
from synth_ai.sdk.api.train.graphgen import GraphGenJob, GraphGenJobResult, GraphGenSubmitResult
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenGoldOutput,
    GraphGenJobConfig,
    GraphGenRubric,
    GraphGenTask,
    GraphGenTaskSet,
    GraphGenVerifierConfig,
    load_graphgen_taskset,
    parse_graphgen_taskset,
)

# Re-export from existing locations
from synth_ai.sdk.api.train.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)

__all__ = [
    # Prompt Learning
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    # GraphGen (preferred names)
    "GraphGenJobConfig",
    "GraphGenTaskSet",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenVerifierConfig",
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
    "GraphGenVerifierConfig",
    "load_graphgen_taskset",
    "parse_graphgen_taskset",
]
