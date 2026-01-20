"""Training SDK - prompt learning and Graph Evolve jobs.

This module provides high-level APIs for running training jobs:
- PromptLearningJob: GEPA prompt optimization
- GraphEvolveJob: Graph Opt (simplified workflows API)

Note: SFT and RL training APIs have been moved to the research repo.

Example:
    from synth_ai.sdk.training import PromptLearningJob, GraphEvolveJob

    # Prompt optimization
    job = PromptLearningJob.from_config("config.toml")
    job.submit()
    result = job.poll_until_complete()

    # Graph Evolve workflow optimization (no task app needed)
    graph_job = GraphEvolveJob.from_dataset("my_tasks.json", rollout_budget=100)
    graph_job.submit()
    result = graph_job.stream_until_complete()
"""

from __future__ import annotations

# Graph Evolve (Graph Opt)
from synth_ai.sdk.api.train.graph_evolve import (
    GraphEvolveGoldOutput,
    GraphEvolveJob,
    GraphEvolveJobConfig,
    GraphEvolveJobResult,
    GraphEvolveSubmitResult,
    GraphEvolveTask,
    GraphEvolveTaskSet,
)
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenRubric,
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
    # Graph Evolve
    "GraphEvolveJob",
    "GraphEvolveJobConfig",
    "GraphEvolveJobResult",
    "GraphEvolveSubmitResult",
    "GraphEvolveTaskSet",
    "GraphEvolveTask",
    "GraphEvolveGoldOutput",
    "GraphGenRubric",
    "GraphGenVerifierConfig",
    "load_graphgen_taskset",
    "parse_graphgen_taskset",
]
