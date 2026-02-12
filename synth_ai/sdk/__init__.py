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
    from synth_ai.core.auth import get_or_mint_synth_api_key
    from synth_ai.core.errors import PaymentRequiredError
    from synth_ai.sdk.environment_pools import (
        EnvironmentPoolsClient,
        PlanGatingError,
        PoolTask,
        cancel_rollout,
        create_pool,
        create_pool_task,
        create_rollout,
        create_rollouts_batch,
        delete_pool,
        delete_pool_task,
        download_artifacts_zip,
        fetch_artifact,
        get_capabilities,
        get_openapi_schema,
        get_pool,
        get_pool_metrics,
        get_queue_status,
        get_rollout,
        get_rollout_summary,
        get_rollout_support_bundle,
        get_rollout_usage,
        get_schema_json,
        list_pool_tasks,
        list_pools,
        list_rollout_artifacts,
        list_rollouts,
        replay_rollout,
        stream_rollout_events,
        update_pool,
        update_pool_task,
        validate_rollout,
    )
    from synth_ai.sdk.eval import EvalJob, EvalJobConfig
    from synth_ai.sdk.graphs import GraphCompletionsClient, GraphTarget, VerifierClient
    from synth_ai.sdk.graphs.verifier_schemas import VerifierOptions, VerifierScoreResponse
    from synth_ai.sdk.inference import (
        InferenceArtifactSpec,
        InferenceClient,
        InferenceJobRequest,
        InferenceJobsClient,
        create_inference_job,
        create_inference_job_from_path,
        download_inference_artifact,
        get_inference_job,
    )
    from synth_ai.sdk.localapi import (
        InProcessTaskApp,
        LocalAPIClient,
        LocalAPIConfig,
        TaskAppConfig,
        create_task_app,
    )
    from synth_ai.sdk.localapi._impl import (
        InProcessJobResult,
        merge_dot_overrides,
        resolve_backend_api_base,
        run_in_process_job,
        run_in_process_job_sync,
    )
    from synth_ai.sdk.managed_pools import (
        create_managed_pool_s3_data_source,
        create_managed_pool_upload_data_source,
        create_managed_pool_upload_url,
        upload_managed_pool_bytes,
        upload_managed_pool_file,
    )
    from synth_ai.sdk.optimization import (
        GraphOptimizationJob,
        PolicyOptimizationJob,
    )
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenGoldOutput as GraphEvolveGoldOutput,
    )
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenJobConfig as GraphEvolveJobConfig,
    )
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenRubric,
        GraphGenVerifierConfig,
        load_graphgen_taskset,
    )
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenTask as GraphEvolveTask,
    )
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenTaskSet as GraphEvolveTaskSet,
    )
    from synth_ai.sdk.task_apps import (
        TaskApp as TaskAppModel,
    )
    from synth_ai.sdk.task_apps import (
        TaskAppsClient,
        TaskAppSpec,
        TaskAppType,
    )

    # Legacy aliases
    PromptLearningJob = PolicyOptimizationJob
    PromptLearningJobConfig = None  # Deprecated - use PolicyOptimizationJob.from_config()
    GraphEvolveJob = GraphOptimizationJob
    from synth_ai.core.tunnels import (
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
    # Optimization (new canonical)
    "PolicyOptimizationJob",
    "GraphOptimizationJob",
    # Legacy aliases (for backward compat)
    "PromptLearningJob",  # -> PolicyOptimizationJob
    "GraphEvolveJob",  # -> GraphOptimizationJob
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
    "InferenceJobsClient",
    "InferenceArtifactSpec",
    "InferenceJobRequest",
    "create_inference_job",
    "create_inference_job_from_path",
    "get_inference_job",
    "download_inference_artifact",
    # Auth helpers
    "get_or_mint_synth_api_key",
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
    "create_managed_pool_upload_url",
    "upload_managed_pool_bytes",
    "upload_managed_pool_file",
    "create_managed_pool_upload_data_source",
    "create_managed_pool_s3_data_source",
    # Environment Pools
    "create_rollout",
    "validate_rollout",
    "list_rollouts",
    "create_rollouts_batch",
    "get_rollout",
    "get_rollout_summary",
    "get_rollout_usage",
    "get_rollout_support_bundle",
    "replay_rollout",
    "stream_rollout_events",
    "list_rollout_artifacts",
    "download_artifacts_zip",
    "fetch_artifact",
    "cancel_rollout",
    "list_pools",
    "get_pool",
    "create_pool",
    "update_pool",
    "delete_pool",
    "get_pool_metrics",
    "list_pool_tasks",
    "create_pool_task",
    "update_pool_task",
    "delete_pool_task",
    "get_queue_status",
    "get_capabilities",
    "get_openapi_schema",
    "get_schema_json",
    "EnvironmentPoolsClient",
    "PoolTask",
    "PlanGatingError",
    "PaymentRequiredError",
    # Hosted Task Apps
    "TaskAppsClient",
    "TaskAppSpec",
    "TaskAppModel",
    "TaskAppType",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "InferenceClient": ("synth_ai.sdk.inference", "InferenceClient"),
    "InferenceJobsClient": ("synth_ai.sdk.inference", "InferenceJobsClient"),
    "InferenceArtifactSpec": ("synth_ai.sdk.inference", "InferenceArtifactSpec"),
    "InferenceJobRequest": ("synth_ai.sdk.inference", "InferenceJobRequest"),
    "create_inference_job": ("synth_ai.sdk.inference", "create_inference_job"),
    "create_inference_job_from_path": ("synth_ai.sdk.inference", "create_inference_job_from_path"),
    "get_inference_job": ("synth_ai.sdk.inference", "get_inference_job"),
    "download_inference_artifact": (
        "synth_ai.sdk.inference",
        "download_inference_artifact",
    ),
    "EvalJob": ("synth_ai.sdk.eval", "EvalJob"),
    "EvalJobConfig": ("synth_ai.sdk.eval", "EvalJobConfig"),
    "get_or_mint_synth_api_key": ("synth_ai.core.auth", "get_or_mint_synth_api_key"),
    "GraphCompletionsClient": ("synth_ai.sdk.graphs", "GraphCompletionsClient"),
    "GraphTarget": ("synth_ai.sdk.graphs", "GraphTarget"),
    "VerifierClient": ("synth_ai.sdk.graphs", "VerifierClient"),
    "VerifierOptions": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierOptions"),
    "VerifierScoreResponse": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierScoreResponse"),
    "InProcessJobResult": ("synth_ai.sdk.localapi._impl", "InProcessJobResult"),
    "InProcessTaskApp": ("synth_ai.sdk.localapi", "InProcessTaskApp"),
    "LocalAPIClient": ("synth_ai.sdk.localapi", "LocalAPIClient"),
    "LocalAPIConfig": ("synth_ai.sdk.localapi", "LocalAPIConfig"),
    "TaskAppConfig": ("synth_ai.sdk.localapi._impl.server", "TaskAppConfig"),
    "create_task_app": ("synth_ai.sdk.localapi", "create_task_app"),
    "merge_dot_overrides": ("synth_ai.sdk.localapi._impl", "merge_dot_overrides"),
    "resolve_backend_api_base": ("synth_ai.sdk.localapi._impl", "resolve_backend_api_base"),
    "run_in_process_job": ("synth_ai.sdk.localapi._impl", "run_in_process_job"),
    "run_in_process_job_sync": ("synth_ai.sdk.localapi._impl", "run_in_process_job_sync"),
    "GraphEvolveGoldOutput": (
        "synth_ai.sdk.optimization.internal.graphgen_models",
        "GraphGenGoldOutput",
    ),
    "GraphEvolveJob": ("synth_ai.sdk.optimization", "GraphOptimizationJob"),
    "GraphEvolveJobConfig": (
        "synth_ai.sdk.optimization.internal.graphgen_models",
        "GraphGenJobConfig",
    ),
    "GraphEvolveTask": ("synth_ai.sdk.optimization.internal.graphgen_models", "GraphGenTask"),
    "GraphEvolveTaskSet": ("synth_ai.sdk.optimization.internal.graphgen_models", "GraphGenTaskSet"),
    "GraphGenRubric": ("synth_ai.sdk.optimization.internal.graphgen_models", "GraphGenRubric"),
    "GraphGenVerifierConfig": (
        "synth_ai.sdk.optimization.internal.graphgen_models",
        "GraphGenVerifierConfig",
    ),
    "GraphOptimizationJob": ("synth_ai.sdk.optimization", "GraphOptimizationJob"),
    "PolicyOptimizationJob": ("synth_ai.sdk.optimization", "PolicyOptimizationJob"),
    "PromptLearningJob": ("synth_ai.sdk.optimization", "PolicyOptimizationJob"),
    "load_graphgen_taskset": (
        "synth_ai.sdk.optimization.internal.graphgen_models",
        "load_graphgen_taskset",
    ),
    "cleanup_all": ("synth_ai.core.tunnels", "cleanup_all"),
    "find_available_port": ("synth_ai.core.tunnels", "find_available_port"),
    "is_port_available": ("synth_ai.core.tunnels", "is_port_available"),
    "kill_port": ("synth_ai.core.tunnels", "kill_port"),
    "open_managed_tunnel": ("synth_ai.core.tunnels", "open_managed_tunnel"),
    "rotate_tunnel": ("synth_ai.core.tunnels", "rotate_tunnel"),
    "stop_tunnel": ("synth_ai.core.tunnels", "stop_tunnel"),
    "track_process": ("synth_ai.core.tunnels", "track_process"),
    "verify_tunnel_dns_resolution": ("synth_ai.core.tunnels", "verify_tunnel_dns_resolution"),
    "wait_for_health_check": ("synth_ai.core.tunnels", "wait_for_health_check"),
    "create_managed_pool_upload_url": (
        "synth_ai.sdk.managed_pools",
        "create_managed_pool_upload_url",
    ),
    "upload_managed_pool_bytes": (
        "synth_ai.sdk.managed_pools",
        "upload_managed_pool_bytes",
    ),
    "upload_managed_pool_file": (
        "synth_ai.sdk.managed_pools",
        "upload_managed_pool_file",
    ),
    "create_managed_pool_upload_data_source": (
        "synth_ai.sdk.managed_pools",
        "create_managed_pool_upload_data_source",
    ),
    "create_managed_pool_s3_data_source": (
        "synth_ai.sdk.managed_pools",
        "create_managed_pool_s3_data_source",
    ),
    "create_rollout": ("synth_ai.sdk.environment_pools", "create_rollout"),
    "validate_rollout": ("synth_ai.sdk.environment_pools", "validate_rollout"),
    "list_rollouts": ("synth_ai.sdk.environment_pools", "list_rollouts"),
    "create_rollouts_batch": ("synth_ai.sdk.environment_pools", "create_rollouts_batch"),
    "get_rollout": ("synth_ai.sdk.environment_pools", "get_rollout"),
    "get_rollout_summary": ("synth_ai.sdk.environment_pools", "get_rollout_summary"),
    "get_rollout_usage": ("synth_ai.sdk.environment_pools", "get_rollout_usage"),
    "get_rollout_support_bundle": (
        "synth_ai.sdk.environment_pools",
        "get_rollout_support_bundle",
    ),
    "replay_rollout": ("synth_ai.sdk.environment_pools", "replay_rollout"),
    "stream_rollout_events": ("synth_ai.sdk.environment_pools", "stream_rollout_events"),
    "list_rollout_artifacts": ("synth_ai.sdk.environment_pools", "list_rollout_artifacts"),
    "download_artifacts_zip": ("synth_ai.sdk.environment_pools", "download_artifacts_zip"),
    "fetch_artifact": ("synth_ai.sdk.environment_pools", "fetch_artifact"),
    "cancel_rollout": ("synth_ai.sdk.environment_pools", "cancel_rollout"),
    "list_pools": ("synth_ai.sdk.environment_pools", "list_pools"),
    "get_pool": ("synth_ai.sdk.environment_pools", "get_pool"),
    "create_pool": ("synth_ai.sdk.environment_pools", "create_pool"),
    "update_pool": ("synth_ai.sdk.environment_pools", "update_pool"),
    "delete_pool": ("synth_ai.sdk.environment_pools", "delete_pool"),
    "get_pool_metrics": ("synth_ai.sdk.environment_pools", "get_pool_metrics"),
    "list_pool_tasks": ("synth_ai.sdk.environment_pools", "list_pool_tasks"),
    "create_pool_task": ("synth_ai.sdk.environment_pools", "create_pool_task"),
    "update_pool_task": ("synth_ai.sdk.environment_pools", "update_pool_task"),
    "delete_pool_task": ("synth_ai.sdk.environment_pools", "delete_pool_task"),
    "get_queue_status": ("synth_ai.sdk.environment_pools", "get_queue_status"),
    "get_capabilities": ("synth_ai.sdk.environment_pools", "get_capabilities"),
    "get_openapi_schema": ("synth_ai.sdk.environment_pools", "get_openapi_schema"),
    "get_schema_json": ("synth_ai.sdk.environment_pools", "get_schema_json"),
    "EnvironmentPoolsClient": ("synth_ai.sdk.environment_pools", "EnvironmentPoolsClient"),
    "PoolResponse": ("synth_ai.sdk.environment_pools", "PoolResponse"),
    "PoolTask": ("synth_ai.sdk.environment_pools", "PoolTask"),
    "PoolTemplate": ("synth_ai.sdk.environment_pools", "PoolTemplate"),
    "PlanGatingError": ("synth_ai.core.errors", "PlanGatingError"),
    "PaymentRequiredError": ("synth_ai.core.errors", "PaymentRequiredError"),
    "TaskAppsClient": ("synth_ai.sdk.task_apps", "TaskAppsClient"),
    "TaskAppSpec": ("synth_ai.sdk.task_apps", "TaskAppSpec"),
    "TaskAppModel": ("synth_ai.sdk.task_apps", "TaskApp"),
    "TaskAppType": ("synth_ai.sdk.task_apps", "TaskAppType"),
}

_OPTIONAL_EXPORTS: set[str] = set()


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
