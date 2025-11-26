"""Synth AI SDK Layer.

This module provides the user-facing programmatic API for:
- Training (prompt learning, SFT, RL)
- Task apps (in-process, deployed, Modal)
- Judging (LLM-based evaluation)
- Inference (model inference via Synth)
- Tracing (session traces)
- Specs (system specifications)
- Research agents (scaffold tuning, evaluation)

Usage:
    from synth_ai.sdk import (
        PromptLearningJob,
        InProcessTaskApp,
        JudgeClient,
        InferenceClient,
    )

Dependency rules:
- sdk/ can import data/ and core/
- sdk/ should NOT import cli/
"""

from __future__ import annotations

# Research Agent
from synth_ai.sdk.api.research_agent import ResearchAgentJob, ResearchAgentJobConfig

# Inference
from synth_ai.sdk.inference import InferenceClient

# Jobs API Client
from synth_ai.sdk.jobs import JobsClient

# Judging
from synth_ai.sdk.judging import JudgeClient, JudgeOptions, JudgeScoreResponse

# Specs
from synth_ai.sdk.specs import (
    load_spec_from_dict,
    load_spec_from_file,
    spec_to_prompt_context,
    validate_spec_dict,
    validate_spec_file,
)

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
from synth_ai.sdk.training import PromptLearningJob, PromptLearningJobConfig, SFTJob

__all__ = [
    # Training
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "SFTJob",
    # Task Apps
    "InProcessTaskApp",
    "InProcessJobResult",
    "merge_dot_overrides",
    "resolve_backend_api_base",
    "run_in_process_job",
    "run_in_process_job_sync",
    "TaskAppConfig",
    "create_task_app",
    # Judging
    "JudgeClient",
    "JudgeOptions",
    "JudgeScoreResponse",
    # Inference
    "InferenceClient",
    # Specs
    "load_spec_from_dict",
    "load_spec_from_file",
    "spec_to_prompt_context",
    "validate_spec_dict",
    "validate_spec_file",
    # Research Agent
    "ResearchAgentJob",
    "ResearchAgentJobConfig",
    # Jobs API Client
    "JobsClient",
]
