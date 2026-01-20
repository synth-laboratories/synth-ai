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

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
    from synth_ai.sdk.auth import get_or_mint_synth_api_key
    from synth_ai.sdk.graphs import GraphCompletionsClient, GraphTarget, VerifierClient
    from synth_ai.sdk.graphs.verifier_schemas import VerifierOptions, VerifierScoreResponse
    from synth_ai.sdk.inference import InferenceClient
    from synth_ai.sdk.jobs import JobsClient
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
    from synth_ai.sdk.training import (
        GraphEvolveGoldOutput,
        GraphEvolveJob,
        GraphEvolveJobConfig,
        GraphEvolveTask,
        GraphEvolveTaskSet,
        GraphGenRubric,
        GraphGenVerifierConfig,
        PromptLearningJob,
        PromptLearningJobConfig,
        load_graphgen_taskset,
    )
    from synth_ai.sdk.tunnels import (
        cleanup_all,
        find_available_port,
        is_port_available,
        kill_port,
        open_managed_tunnel,
        rotate_tunnel,
        stop_tunnel,
        track_process,
        verify_tunnel_dns_resolution,
        wait_for_health_check,
    )


__all__ = [
    # Training
    "PromptLearningJob",
    "PromptLearningJobConfig",
    # Graph Evolve
    "GraphEvolveJob",
    "GraphEvolveJobConfig",
    "GraphEvolveTaskSet",
    "GraphEvolveTask",
    "GraphEvolveGoldOutput",
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
    # Auth helpers
    "get_or_mint_synth_api_key",
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

_EXPORTS: dict[str, tuple[str, str]] = {
    "InferenceClient": ("synth_ai.sdk.inference", "InferenceClient"),
    "JobsClient": ("synth_ai.sdk.jobs", "JobsClient"),
    "EvalJob": ("synth_ai.sdk.api.eval", "EvalJob"),
    "EvalJobConfig": ("synth_ai.sdk.api.eval", "EvalJobConfig"),
    "get_or_mint_synth_api_key": ("synth_ai.sdk.auth", "get_or_mint_synth_api_key"),
    "GraphCompletionsClient": ("synth_ai.sdk.graphs", "GraphCompletionsClient"),
    "GraphTarget": ("synth_ai.sdk.graphs", "GraphTarget"),
    "VerifierClient": ("synth_ai.sdk.graphs", "VerifierClient"),
    "VerifierOptions": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierOptions"),
    "VerifierScoreResponse": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierScoreResponse"),
    "InProcessJobResult": ("synth_ai.sdk.task", "InProcessJobResult"),
    "InProcessTaskApp": ("synth_ai.sdk.task", "InProcessTaskApp"),
    "LocalAPIClient": ("synth_ai.sdk.task", "LocalAPIClient"),
    "LocalAPIConfig": ("synth_ai.sdk.task", "LocalAPIConfig"),
    "TaskAppConfig": ("synth_ai.sdk.task", "TaskAppConfig"),
    "create_task_app": ("synth_ai.sdk.task", "create_task_app"),
    "merge_dot_overrides": ("synth_ai.sdk.task", "merge_dot_overrides"),
    "resolve_backend_api_base": ("synth_ai.sdk.task", "resolve_backend_api_base"),
    "run_in_process_job": ("synth_ai.sdk.task", "run_in_process_job"),
    "run_in_process_job_sync": ("synth_ai.sdk.task", "run_in_process_job_sync"),
    "GraphEvolveGoldOutput": ("synth_ai.sdk.training", "GraphEvolveGoldOutput"),
    "GraphEvolveJob": ("synth_ai.sdk.training", "GraphEvolveJob"),
    "GraphEvolveJobConfig": ("synth_ai.sdk.training", "GraphEvolveJobConfig"),
    "GraphEvolveTask": ("synth_ai.sdk.training", "GraphEvolveTask"),
    "GraphEvolveTaskSet": ("synth_ai.sdk.training", "GraphEvolveTaskSet"),
    "GraphGenRubric": ("synth_ai.sdk.training", "GraphGenRubric"),
    "GraphGenVerifierConfig": ("synth_ai.sdk.training", "GraphGenVerifierConfig"),
    "PromptLearningJob": ("synth_ai.sdk.training", "PromptLearningJob"),
    "PromptLearningJobConfig": ("synth_ai.sdk.training", "PromptLearningJobConfig"),
    "load_graphgen_taskset": ("synth_ai.sdk.training", "load_graphgen_taskset"),
    "cleanup_all": ("synth_ai.sdk.tunnels", "cleanup_all"),
    "find_available_port": ("synth_ai.sdk.tunnels", "find_available_port"),
    "is_port_available": ("synth_ai.sdk.tunnels", "is_port_available"),
    "kill_port": ("synth_ai.sdk.tunnels", "kill_port"),
    "open_managed_tunnel": ("synth_ai.sdk.tunnels", "open_managed_tunnel"),
    "rotate_tunnel": ("synth_ai.sdk.tunnels", "rotate_tunnel"),
    "stop_tunnel": ("synth_ai.sdk.tunnels", "stop_tunnel"),
    "track_process": ("synth_ai.sdk.tunnels", "track_process"),
    "verify_tunnel_dns_resolution": ("synth_ai.sdk.tunnels", "verify_tunnel_dns_resolution"),
    "wait_for_health_check": ("synth_ai.sdk.tunnels", "wait_for_health_check"),
}

_OPTIONAL_EXPORTS = {"JobsClient"}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        if name in _OPTIONAL_EXPORTS:
            return None
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(module, attr_name)
