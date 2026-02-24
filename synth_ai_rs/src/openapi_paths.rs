//! Canonical OpenAPI path registry for the Rust SDK.
//! Keep these constants aligned with `openapi/synth-api-v1.yaml`.

pub const API_GRAPHS_COMPLETIONS: &str = "/api/graphs/completions";
pub const API_INFERENCE_JOBS: &str = "/api/inference/jobs";
pub const API_INFERENCE_JOBS_JOB_ID: &str = "/api/inference/jobs/{job_id}";
pub const API_INFERENCE_CHAT_COMPLETIONS: &str = "/api/inference/v1/chat/completions";
pub const API_V1_CONTAINERS: &str = "/api/v1/containers";
pub const API_V1_CONTAINERS_CONTAINER_ID: &str = "/api/v1/containers/{container_id}";
pub const API_V1_SYNTHTUNNEL_LEASES: &str = "/api/v1/synthtunnel/leases";
pub const API_V1_SYNTHTUNNEL_LEASES_LEASE_ID: &str = "/api/v1/synthtunnel/leases/{lease_id}";
pub const GRAPH_EVOLVE_GRAPHS: &str = "/graph-evolve/graphs";
pub const INFERENCE_JOBS_JOB_ID_ARTIFACTS_ARTIFACT_ID: &str =
    "/inference/jobs/{job_id}/artifacts/{artifact_id}";

pub const V1_OFFLINE_JOBS: &str = "/v1/offline/jobs";
pub const V1_OFFLINE_JOBS_JOB_ID: &str = "/v1/offline/jobs/{job_id}";
pub const V1_OFFLINE_JOBS_JOB_ID_ARTIFACTS: &str = "/v1/offline/jobs/{job_id}/artifacts";
pub const V1_OFFLINE_JOBS_JOB_ID_EVENTS: &str = "/v1/offline/jobs/{job_id}/events";

pub const V1_ONLINE_SESSIONS: &str = "/v1/online/sessions";
pub const V1_ONLINE_SESSIONS_SESSION_ID: &str = "/v1/online/sessions/{session_id}";
pub const V1_ONLINE_SESSIONS_SESSION_ID_REWARD: &str = "/v1/online/sessions/{session_id}/reward";
pub const V1_ONLINE_SESSIONS_SESSION_ID_EVENTS: &str = "/v1/online/sessions/{session_id}/events";

pub const V1_POLICY_OPTIMIZATION_SYSTEMS: &str = "/v1/policy-optimization/systems";
pub const V1_POLICY_OPTIMIZATION_SYSTEMS_SYSTEM_ID: &str =
    "/v1/policy-optimization/systems/{system_id}";

pub const V1_POOLS: &str = "/v1/pools";
pub const V1_POOLS_ASSEMBLIES: &str = "/v1/pools/assemblies";
pub const V1_POOLS_ASSEMBLIES_ASSEMBLY_ID: &str = "/v1/pools/assemblies/{assembly_id}";
pub const V1_POOLS_ASSEMBLIES_ASSEMBLY_ID_EVENTS: &str =
    "/v1/pools/assemblies/{assembly_id}/events";
pub const V1_POOLS_DATA_SOURCES: &str = "/v1/pools/data-sources";
pub const V1_POOLS_DATA_SOURCES_DATA_SOURCE_ID: &str = "/v1/pools/data-sources/{data_source_id}";
pub const V1_POOLS_DATA_SOURCES_DATA_SOURCE_ID_REFRESH: &str =
    "/v1/pools/data-sources/{data_source_id}/refresh";
pub const V1_POOLS_UPLOADS: &str = "/v1/pools/uploads";
pub const V1_POOLS_UPLOADS_UPLOAD_ID: &str = "/v1/pools/uploads/{upload_id}";
pub const V1_POOLS_POOL_ID: &str = "/v1/pools/{pool_id}";
pub const V1_POOLS_POOL_ID_ASSEMBLIES: &str = "/v1/pools/{pool_id}/assemblies";
pub const V1_POOLS_POOL_ID_METRICS: &str = "/v1/pools/{pool_id}/metrics";
pub const V1_POOLS_POOL_ID_ROLLOUTS: &str = "/v1/pools/{pool_id}/rollouts";
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID: &str =
    "/v1/pools/{pool_id}/rollouts/{rollout_id}";
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID_ARTIFACTS: &str =
    "/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts";
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID_CANCEL: &str =
    "/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel";
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID_EVENTS: &str =
    "/v1/pools/{pool_id}/rollouts/{rollout_id}/events";
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID_USAGE: &str =
    "/v1/pools/{pool_id}/rollouts/{rollout_id}/usage";
pub const V1_POOLS_POOL_ID_TASKS: &str = "/v1/pools/{pool_id}/tasks";
pub const V1_POOLS_POOL_ID_TASKS_TASK_ID: &str = "/v1/pools/{pool_id}/tasks/{task_id}";

pub const V1_ROLLOUTS: &str = "/v1/rollouts";
pub const V1_ROLLOUTS_ROLLOUT_ID: &str = "/v1/rollouts/{rollout_id}";
pub const V1_ROLLOUTS_ROLLOUT_ID_ARTIFACTS: &str = "/v1/rollouts/{rollout_id}/artifacts";
pub const V1_ROLLOUTS_ROLLOUT_ID_ARTIFACTS_ZIP: &str = "/v1/rollouts/{rollout_id}/artifacts.zip";
pub const V1_ROLLOUTS_ROLLOUT_ID_ARTIFACTS_PATH: &str =
    "/v1/rollouts/{rollout_id}/artifacts/{path}";
pub const V1_ROLLOUTS_ROLLOUT_ID_SUPPORT_BUNDLE: &str = "/v1/rollouts/{rollout_id}/support_bundle";

pub fn api_inference_jobs_job(job_id: &str) -> String {
    format!("/api/inference/jobs/{job_id}")
}

pub fn api_v1_container(container_id: &str) -> String {
    format!("/api/v1/containers/{container_id}")
}

pub fn api_v1_synthtunnel_lease(lease_id: &str) -> String {
    format!("/api/v1/synthtunnel/leases/{lease_id}")
}

pub fn inference_job_artifact(job_id: &str, artifact_id: &str) -> String {
    format!("/inference/jobs/{job_id}/artifacts/{artifact_id}")
}

pub fn v1_offline_job(job_id: &str) -> String {
    format!("/v1/offline/jobs/{job_id}")
}

pub fn v1_offline_job_artifacts(job_id: &str) -> String {
    format!("/v1/offline/jobs/{job_id}/artifacts")
}

pub fn v1_offline_job_events(job_id: &str) -> String {
    format!("/v1/offline/jobs/{job_id}/events")
}

pub fn v1_online_session(session_id: &str) -> String {
    format!("/v1/online/sessions/{session_id}")
}

pub fn v1_online_session_reward(session_id: &str) -> String {
    format!("/v1/online/sessions/{session_id}/reward")
}

pub fn v1_online_session_events(session_id: &str) -> String {
    format!("/v1/online/sessions/{session_id}/events")
}

pub fn v1_policy_optimization_system(system_id: &str) -> String {
    format!("/v1/policy-optimization/systems/{system_id}")
}

pub fn v1_pools_upload(upload_id: &str) -> String {
    format!("/v1/pools/uploads/{upload_id}")
}

pub fn v1_pools_data_source(data_source_id: &str) -> String {
    format!("/v1/pools/data-sources/{data_source_id}")
}

pub fn v1_pools_data_source_refresh(data_source_id: &str) -> String {
    format!("/v1/pools/data-sources/{data_source_id}/refresh")
}

pub fn v1_pools_assembly(assembly_id: &str) -> String {
    format!("/v1/pools/assemblies/{assembly_id}")
}

pub fn v1_pools_assembly_events(assembly_id: &str) -> String {
    format!("/v1/pools/assemblies/{assembly_id}/events")
}

pub fn v1_pool(pool_id: &str) -> String {
    format!("/v1/pools/{pool_id}")
}

pub fn v1_pool_assemblies(pool_id: &str) -> String {
    format!("/v1/pools/{pool_id}/assemblies")
}

pub fn v1_pool_metrics(pool_id: &str) -> String {
    format!("/v1/pools/{pool_id}/metrics")
}

pub fn v1_pool_rollouts(pool_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts")
}

pub fn v1_pool_rollout(pool_id: &str, rollout_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts/{rollout_id}")
}

pub fn v1_pool_rollout_artifacts(pool_id: &str, rollout_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")
}

pub fn v1_pool_rollout_cancel(pool_id: &str, rollout_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel")
}

pub fn v1_pool_rollout_events(pool_id: &str, rollout_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts/{rollout_id}/events")
}

pub fn v1_pool_rollout_usage(pool_id: &str, rollout_id: &str) -> String {
    format!("/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")
}

pub fn v1_pool_tasks(pool_id: &str) -> String {
    format!("/v1/pools/{pool_id}/tasks")
}

pub fn v1_pool_task(pool_id: &str, task_id: &str) -> String {
    format!("/v1/pools/{pool_id}/tasks/{task_id}")
}

pub fn v1_rollout(rollout_id: &str) -> String {
    format!("/v1/rollouts/{rollout_id}")
}

pub fn v1_rollout_artifacts(rollout_id: &str) -> String {
    format!("/v1/rollouts/{rollout_id}/artifacts")
}

pub fn v1_rollout_artifacts_zip(rollout_id: &str) -> String {
    format!("/v1/rollouts/{rollout_id}/artifacts.zip")
}

pub fn v1_rollout_artifact_path(rollout_id: &str, path: &str) -> String {
    format!("/v1/rollouts/{rollout_id}/artifacts/{path}")
}

pub fn v1_rollout_support_bundle(rollout_id: &str) -> String {
    format!("/v1/rollouts/{rollout_id}/support_bundle")
}
