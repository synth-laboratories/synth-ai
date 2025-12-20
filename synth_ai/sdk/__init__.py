"""Synth AI SDK Layer.

This module provides the user-facing programmatic API for:
- Training (prompt learning, SFT, RL, graph generation)
- Task apps (in-process, deployed, Modal)
- Graphs (verifiers, completions)
- Inference (model inference via Synth)

Usage:
    from synth_ai.sdk import (
        PromptLearningJob,
        InProcessTaskApp,
        VerifierClient,
        InferenceClient,
    )

Dependency rules:
- sdk/ can import data/ and core/
- sdk/ should NOT import cli/
"""

from __future__ import annotations

# Inference
from synth_ai.sdk.inference import InferenceClient

# Jobs API Client
from synth_ai.sdk.jobs import JobsClient

# Judging types and graph clients
from synth_ai.sdk.judging import JudgeOptions, JudgeScoreResponse
from synth_ai.sdk.graphs import GraphCompletionsClient, GraphTarget, VerifierClient

# Task Apps
from synth_ai.sdk.task import (
    InProcessJobResult,
    InProcessTaskApp,
    TaskAppConfig,
    create_task_app,
    merge_dot_overrides,
    resolve_backend_api_base,
    run_in_process_job,
    run_in_process_job_sync,
)

# Training
from synth_ai.sdk.training import (
    PromptLearningJob,
    PromptLearningJobConfig,
    SFTJob,
    GraphGenJob,
    GraphGenJobConfig,
    GraphGenTaskSet,
    GraphGenTask,
    GraphGenGoldOutput,
    GraphGenRubric,
    GraphGenJudgeConfig,
    load_graphgen_taskset,
)

__all__ = [
    # Training
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "SFTJob",
    # GraphGen
    "GraphGenJob",
    "GraphGenJobConfig",
    "GraphGenTaskSet",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenJudgeConfig",
    "load_graphgen_taskset",
    # Task Apps
    "InProcessTaskApp",
    "InProcessJobResult",
    "merge_dot_overrides",
    "resolve_backend_api_base",
    "run_in_process_job",
    "run_in_process_job_sync",
    "TaskAppConfig",
    "create_task_app",
    # Graphs / Judging
    "VerifierClient",
    "JudgeOptions",
    "JudgeScoreResponse",
    "GraphCompletionsClient",
    "GraphTarget",
    # Inference
    "InferenceClient",
    # Jobs API Client
    "JobsClient",
]
