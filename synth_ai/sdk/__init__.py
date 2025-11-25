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

# Training
from synth_ai.sdk.training import (
    PromptLearningJob,
    PromptLearningJobConfig,
    SFTJob,
)

# Task Apps
from synth_ai.sdk.task_apps import (
    InProcessTaskApp,
    TaskAppConfig,
    create_task_app,
)

# Judging
from synth_ai.sdk.judging import JudgeClient, JudgeOptions, JudgeScoreResponse

# Inference
from synth_ai.sdk.inference import InferenceClient

# Specs
from synth_ai.sdk.specs import (
    load_spec_from_dict,
    load_spec_from_file,
    spec_to_prompt_context,
    validate_spec_dict,
    validate_spec_file,
)

# Research Agent
from synth_ai.sdk.research_agent import ResearchAgentJob, ResearchAgentJobConfig

# Jobs API Client
from synth_ai.sdk.jobs import JobsClient

__all__ = [
    # Training
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "SFTJob",
    # Task Apps
    "InProcessTaskApp",
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

