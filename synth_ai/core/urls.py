import os

_BACKEND_URL_BASE = os.getenv("SYNTH_BACKEND_URL") or "https://api.usesynth.ai"
_FRONTEND_URL_BASE = os.getenv("SYNTH_FRONTEND_URL") or "https://usesynth.ai"


# =============================================================================
# Frontend URL functions (usesynth.ai)
# =============================================================================


# Base URL helpers (keep at top)


def _frontend_base_url(base_url: str | None = None) -> str:
    return base_url if base_url else _FRONTEND_URL_BASE


def frontend_api_url(path: str, base_url: str | None = None) -> str:
    return f"{_frontend_base_url(base_url)}/api/{path}"


def frontend_url(path: str, base_url: str | None = None) -> str:
    return f"{_frontend_base_url(base_url)}/{path}"


# Frontend endpoints (alphabetical)


def frontend_billing_url(base_url: str | None = None) -> str:
    """Billing settings page - for user redirects."""
    return frontend_url("settings/billing", base_url)


def frontend_device_init_url(base_url: str | None = None) -> str:
    """Auth API endpoint - used for HTTP POST requests."""
    return frontend_api_url("auth/device/init", base_url)


def frontend_device_token_url(base_url: str | None = None) -> str:
    """Auth API endpoint - used for HTTP POST requests."""
    return frontend_api_url("auth/device/token", base_url)


def frontend_usage_url(base_url: str | None = None) -> str:
    """Usage dashboard page - for user redirects."""
    return frontend_url("usage", base_url)


# =============================================================================
# Backend URL functions (api.usesynth.ai)
# =============================================================================


# Base URL helpers (keep at top)


def _synth_api_base(synth_base_url: str | None = None) -> str:
    return f"{_synth_base_url(synth_base_url)}/api"


def _synth_base_url(synth_base_url: str | None = None) -> str:
    return synth_base_url if synth_base_url else _BACKEND_URL_BASE


def local_backend_url(host: str = "localhost", port: int = 8000) -> str:
    return f"http://{host}:{port}"


def synth_api_base(synth_base_url: str | None = None) -> str:
    return _synth_api_base(synth_base_url)


def synth_api_url(path: str, synth_base_url: str | None = None) -> str:
    return f"{_synth_api_base(synth_base_url)}/{path}"


def synth_api_v1_base(synth_base_url: str | None = None) -> str:
    return f"{synth_api_base(synth_base_url)}/v1"


def synth_api_v1_url(path: str, synth_base_url: str | None = None) -> str:
    return f"{synth_api_v1_base(synth_base_url)}/{path}"


def synth_base_url(synth_base_url: str | None = None) -> str:
    return _synth_base_url(synth_base_url)


def synth_interceptor_base(synth_base_url: str | None = None) -> str:
    base = _synth_base_url(synth_base_url)
    return f"{base}/api/interceptor/v1"


def synth_interceptor_url(path: str, synth_base_url: str | None = None) -> str:
    return f"{synth_interceptor_base(synth_base_url)}/{path}"


def synth_research_anthropic_base(synth_base_url: str | None = None) -> str:
    return synth_research_base(synth_base_url)


def synth_research_base(synth_base_url: str | None = None) -> str:
    base = _synth_base_url(synth_base_url)
    return f"{base}/api/synth-research"


def synth_research_openai_base(synth_base_url: str | None = None) -> str:
    return f"{synth_research_base(synth_base_url)}/v1"


# Backend endpoints (alphabetical)


def synth_artifact_model_url(model_id: str, synth_base_url: str | None = None) -> str:
    return synth_api_url(f"artifacts/models/{model_id}", synth_base_url)


def synth_artifact_prompt_url(job_id: str, synth_base_url: str | None = None) -> str:
    return synth_api_url(f"artifacts/prompts/{job_id}", synth_base_url)


def synth_artifacts_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("artifacts", synth_base_url)


def synth_balance_autumn_normalized_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("balance/autumn-normalized", synth_base_url)


def synth_context_learning_best_script_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_job_url(job_id, synth_base_url)}/best-script"


def synth_context_learning_cancel_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_job_url(job_id, synth_base_url)}/cancel"


def synth_context_learning_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_job_url(job_id, synth_base_url)}/events"


def synth_context_learning_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_jobs_url(synth_base_url)}/{job_id}"


def synth_context_learning_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("context-learning/jobs", synth_base_url)


def synth_context_learning_metrics_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_job_url(job_id, synth_base_url)}/metrics"


def synth_context_learning_start_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_context_learning_job_url(job_id, synth_base_url)}/start"


def synth_demo_keys_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("demo/keys", synth_base_url)


def synth_env_keys_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("env-keys", synth_base_url)


def synth_env_keys_verify_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("env-keys/verify", synth_base_url)


def synth_eval_job_results_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_eval_job_url(job_id, synth_base_url)}/results"


def synth_eval_job_traces_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_eval_job_url(job_id, synth_base_url)}/traces"


def synth_eval_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_eval_jobs_url(synth_base_url)}/{job_id}"


def synth_eval_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("eval/jobs", synth_base_url)


def synth_file_url(file_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_files_url(synth_base_url)}/{file_id}"


def synth_files_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("files", synth_base_url)


def synth_graph_evolve_graphs_url(synth_base_url: str | None = None) -> str:
    base = _synth_base_url(synth_base_url)
    return f"{base}/graph-evolve/graphs"


def synth_graph_evolve_job_result_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graph_evolve_job_url(job_id, synth_base_url)}/result"


def synth_graph_evolve_job_save_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graph_evolve_job_url(job_id, synth_base_url)}/save-graph"


def synth_graph_evolve_job_status_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graph_evolve_job_url(job_id, synth_base_url)}/status"


def synth_graph_evolve_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graph_evolve_jobs_url(synth_base_url)}/{job_id}"


def synth_graph_evolve_jobs_url(synth_base_url: str | None = None) -> str:
    base = _synth_base_url(synth_base_url)
    return f"{base}/graph-evolve/jobs"


def synth_graphgen_graph_completions_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("graphgen/graph/completions", synth_base_url)


def synth_graphgen_graph_record_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("graphgen/graph/record", synth_base_url)


def synth_graphgen_job_download_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_job_url(job_id, synth_base_url)}/download"


def synth_graphgen_job_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_job_url(job_id, synth_base_url)}/events"


def synth_graphgen_job_graph_txt_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_job_url(job_id, synth_base_url)}/graph.txt"


def synth_graphgen_job_metrics_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_job_url(job_id, synth_base_url)}/metrics"


def synth_graphgen_job_start_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_job_url(job_id, synth_base_url)}/start"


def synth_graphgen_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_graphgen_jobs_url(synth_base_url)}/{job_id}"


def synth_graphgen_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("graphgen/jobs", synth_base_url)


def synth_graphs_completions_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("graphs/completions", synth_base_url)


def synth_health_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("health", synth_base_url)


def synth_inference_chat_completions_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("inference/v1/chat/completions", synth_base_url)


def synth_learning_exports_hf_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("learning/exports/hf", synth_base_url)


def synth_learning_files_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("learning/files", synth_base_url)


def synth_learning_job_cancel_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_job_url(job_id, synth_base_url)}/cancel"


def synth_learning_job_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_job_url(job_id, synth_base_url)}/events"


def synth_learning_job_metrics_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_job_url(job_id, synth_base_url)}/metrics"


def synth_learning_job_start_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_job_url(job_id, synth_base_url)}/start"


def synth_learning_job_timeline_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_job_url(job_id, synth_base_url)}/timeline"


def synth_learning_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_learning_jobs_url(synth_base_url)}/{job_id}"


def synth_learning_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("learning/jobs", synth_base_url)


def synth_learning_models_on_wasabi_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("learning/models/on-wasabi", synth_base_url)


def synth_learning_models_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("learning/models", synth_base_url)


def synth_me_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("me", synth_base_url)


def synth_model_url(model_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_models_url(synth_base_url)}/{model_id}"


def synth_models_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("models", synth_base_url)


def synth_ontology_health_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("ontology/health", synth_base_url)


def synth_ontology_node_context_url(name: str, synth_base_url: str | None = None) -> str:
    return f"{synth_ontology_node_url(name, synth_base_url)}/context"


def synth_ontology_node_neighborhood_url(name: str, synth_base_url: str | None = None) -> str:
    return f"{synth_ontology_node_url(name, synth_base_url)}/neighborhood"


def synth_ontology_node_properties_url(name: str, synth_base_url: str | None = None) -> str:
    return f"{synth_ontology_node_url(name, synth_base_url)}/properties"


def synth_ontology_node_relationships_incoming_url(
    name: str, synth_base_url: str | None = None
) -> str:
    return f"{synth_ontology_node_url(name, synth_base_url)}/relationships/incoming"


def synth_ontology_node_relationships_outgoing_url(
    name: str, synth_base_url: str | None = None
) -> str:
    return f"{synth_ontology_node_url(name, synth_base_url)}/relationships/outgoing"


def synth_ontology_node_url(name: str, synth_base_url: str | None = None) -> str:
    return f"{synth_ontology_nodes_url(synth_base_url)}/{name}"


def synth_ontology_nodes_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("ontology/nodes", synth_base_url)


def synth_ontology_properties_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("ontology/properties", synth_base_url)


def synth_orchestration_job_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_orchestration_job_url(job_id, synth_base_url)}/events"


def synth_orchestration_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_orchestration_jobs_url(synth_base_url)}/{job_id}"


def synth_orchestration_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("orchestration/jobs", synth_base_url)


def synth_pricing_preflight_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("pricing/preflight", synth_base_url)


def synth_prompt_learning_artifacts_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_prompt_learning_job_url(job_id, synth_base_url)}/artifacts"


def synth_prompt_learning_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_prompt_learning_job_url(job_id, synth_base_url)}/events"


def synth_prompt_learning_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_prompt_learning_jobs_url(synth_base_url)}/{job_id}"


def synth_prompt_learning_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("prompt-learning/online/jobs", synth_base_url)


def synth_prompt_learning_patterns_discover_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("prompt-learning/patterns/discover", synth_base_url)


def synth_prompt_learning_snapshot_url(
    job_id: str, snapshot_id: str, synth_base_url: str | None = None
) -> str:
    return f"{synth_prompt_learning_snapshots_url(job_id, synth_base_url)}/{snapshot_id}"


def synth_prompt_learning_snapshots_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_prompt_learning_job_url(job_id, synth_base_url)}/snapshots"


def synth_public_key_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("crypto/public-key", synth_base_url)


def synth_rl_job_cancel_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_rl_job_url(job_id, synth_base_url)}/cancel"


def synth_rl_job_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_rl_job_url(job_id, synth_base_url)}/events"


def synth_rl_job_metrics_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_rl_job_url(job_id, synth_base_url)}/metrics"


def synth_rl_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_rl_jobs_url(synth_base_url)}/{job_id}"


def synth_rl_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("rl/jobs", synth_base_url)


def synth_rl_verify_task_app_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("rl/verify_task_app", synth_base_url)


def synth_sdk_logs_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("sdk-logs", synth_base_url)


def synth_session_end_url(session_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_session_url(session_id, synth_base_url)}/end"


def synth_session_limits_increase_url(session_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_session_url(session_id, synth_base_url)}/limits/increase"


def synth_session_limits_url(session_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_session_url(session_id, synth_base_url)}/limits"


def synth_session_url(session_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_sessions_url(synth_base_url)}/{session_id}"


def synth_session_usage_url(session_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_session_url(session_id, synth_base_url)}/usage"


def synth_sessions_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("sessions", synth_base_url)


def synth_sft_job_events_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_sft_job_url(job_id, synth_base_url)}/events"


def synth_sft_job_url(job_id: str, synth_base_url: str | None = None) -> str:
    return f"{synth_sft_jobs_url(synth_base_url)}/{job_id}"


def synth_sft_jobs_url(synth_base_url: str | None = None) -> str:
    return synth_api_url("fine_tuning/jobs", synth_base_url)


def synth_tunnel_rotate_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("tunnels/rotate", synth_base_url)


def synth_tunnels_url(synth_base_url: str | None = None) -> str:
    return synth_api_v1_url("tunnels", synth_base_url)
