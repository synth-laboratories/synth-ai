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
try:
    from synth_ai.sdk.jobs import JobsClient
except Exception:  # pragma: no cover - optional dependency guard
    JobsClient = None  # type: ignore[assignment]

# Verifier types and graph clients
from synth_ai.sdk.graphs import GraphCompletionsClient, GraphTarget, VerifierClient
from synth_ai.sdk.graphs.verifier_schemas import VerifierOptions, VerifierScoreResponse

# Task Apps
from synth_ai.sdk.task import (
    InProcessJobResult,
    InProcessTaskApp,
    LocalAPIClient,
    LocalAPIConfig,
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
    GraphGenVerifierConfig,
    load_graphgen_taskset,
)

# Evaluation
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig

# Tunnels - commonly used functions for notebook/script usage
from synth_ai.sdk.tunnels import (
    rotate_tunnel,
    open_managed_tunnel,
    stop_tunnel,
    track_process,
    cleanup_all,
    verify_tunnel_dns_resolution,
    wait_for_health_check,
    kill_port,
    is_port_available,
    find_available_port,
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
    "GraphGenVerifierConfig",
    "load_graphgen_taskset",
    # Evaluation
    "EvalJob",
    "EvalJobConfig",
    # Task Apps
    "InProcessTaskApp",
    "InProcessJobResult",
    "merge_dot_overrides",
    "resolve_backend_api_base",
    "run_in_process_job",
    "run_in_process_job_sync",
    "LocalAPIClient",
    "LocalAPIConfig",
    "TaskAppConfig",
    "create_task_app",
    # Graphs / Verifier
    "VerifierClient",
    "VerifierOptions",
    "VerifierScoreResponse",
    "GraphCompletionsClient",
    "GraphTarget",
    # Inference
    "InferenceClient",
    # Jobs API Client
    "JobsClient",
    # Tunnels
    "rotate_tunnel",
    "open_managed_tunnel",
    "stop_tunnel",
    "track_process",
    "cleanup_all",
    "verify_tunnel_dns_resolution",
    "wait_for_health_check",
    "kill_port",
    "is_port_available",
    "find_available_port",
]
