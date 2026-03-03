//! Canonical OpenAPI path registry for the Rust SDK.
//! Keep these constants aligned with `openapi/synth-api-v1.yaml`.

pub const API_V1_CONTAINERS: &str = "/api/v1/containers";
pub const API_V1_CONTAINERS_CONTAINER_ID: &str = "/api/v1/containers/{container_id}";
pub const API_V1_SYNTHTUNNEL_LEASES: &str = "/api/v1/synthtunnel/leases";
pub const API_V1_SYNTHTUNNEL_LEASES_LEASE_ID: &str = "/api/v1/synthtunnel/leases/{lease_id}";

pub const V1_OFFLINE_JOBS: &str = "/v1/offline/jobs";
pub const V1_OFFLINE_JOBS_JOB_ID: &str = "/v1/offline/jobs/{job_id}";
pub const V1_OFFLINE_JOBS_JOB_ID_ARTIFACTS: &str = "/v1/offline/jobs/{job_id}/artifacts";
pub const V1_OFFLINE_JOBS_JOB_ID_EVENTS: &str = "/v1/offline/jobs/{job_id}/events";

// ---------------------------------------------------------------------------
// V2 offline jobs
// ---------------------------------------------------------------------------

pub const V2_OFFLINE_JOBS: &str = "/v2/offline/jobs";
pub const V2_OFFLINE_JOBS_JOB_ID: &str = "/v2/offline/jobs/{job_id}";
pub const V2_OFFLINE_JOBS_JOB_ID_ARTIFACTS: &str = "/v2/offline/jobs/{job_id}/artifacts";
pub const V2_OFFLINE_JOBS_JOB_ID_EVENTS: &str = "/v2/offline/jobs/{job_id}/events";

pub fn v2_offline_job(job_id: &str) -> String {
    format!("/v2/offline/jobs/{job_id}")
}

pub fn v2_offline_job_artifacts(job_id: &str) -> String {
    format!("/v2/offline/jobs/{job_id}/artifacts")
}

pub fn v2_offline_job_events(job_id: &str) -> String {
    format!("/v2/offline/jobs/{job_id}/events")
}

pub const V1_ONLINE_SESSIONS: &str = "/v1/online/sessions";
pub const V1_ONLINE_SESSIONS_SESSION_ID: &str = "/v1/online/sessions/{session_id}";
pub const V1_ONLINE_SESSIONS_SESSION_ID_REWARD: &str = "/v1/online/sessions/{session_id}/reward";
pub const V1_ONLINE_SESSIONS_SESSION_ID_EVENTS: &str = "/v1/online/sessions/{session_id}/events";

// ---------------------------------------------------------------------------
// V2 online sessions
// ---------------------------------------------------------------------------

pub const V2_ONLINE_SESSIONS: &str = "/v2/online/sessions";
pub const V2_ONLINE_SESSIONS_SESSION_ID: &str = "/v2/online/sessions/{session_id}";
pub const V2_ONLINE_SESSIONS_SESSION_ID_REWARD: &str = "/v2/online/sessions/{session_id}/reward";
pub const V2_ONLINE_SESSIONS_SESSION_ID_EVENTS: &str = "/v2/online/sessions/{session_id}/events";
pub const V2_RUNTIME_COMPATIBILITY: &str = "/v2/runtime/compatibility";
pub const V1_OPTIMIZER_EVENTS: &str = "/v1/optimizer/events";
pub const V2_OPTIMIZER_EVENTS: &str = "/v2/optimizer/events";
pub const V1_FAILURES_QUERY: &str = "/v1/failures/query";
pub const V2_FAILURES_QUERY: &str = "/v2/failures/query";
pub const V1_ADMIN_OPTIMIZER_EVENTS: &str = "/v1/admin/optimizer/events";
pub const V2_ADMIN_OPTIMIZER_EVENTS: &str = "/v2/admin/optimizer/events";
pub const V1_ADMIN_FAILURES_QUERY: &str = "/v1/admin/failures/query";
pub const V2_ADMIN_FAILURES_QUERY: &str = "/v2/admin/failures/query";
pub const V1_ADMIN_VICTORIA_LOGS_QUERY: &str = "/v1/admin/victoria-logs/query";
pub const V2_ADMIN_VICTORIA_LOGS_QUERY: &str = "/v2/admin/victoria-logs/query";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_TRIALS: &str =
    "/v2/runtime/systems/{system_id}/queue/trials";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_CONTRACT: &str =
    "/v2/runtime/systems/{system_id}/queue/contract";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_TRIALS_TRIAL_ID: &str =
    "/v2/runtime/systems/{system_id}/queue/trials/{trial_id}";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_ROLLOUTS: &str =
    "/v2/runtime/systems/{system_id}/queue/rollouts";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_ROLLOUTS_LEASE: &str =
    "/v2/runtime/systems/{system_id}/queue/rollouts/lease";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_ROLLOUTS_EXPIRE_LEASES: &str =
    "/v2/runtime/systems/{system_id}/queue/rollouts/expire-leases";
pub const V2_RUNTIME_SYSTEMS_SYSTEM_ID_QUEUE_ROLLOUTS_ROLLOUT_ID: &str =
    "/v2/runtime/systems/{system_id}/queue/rollouts/{rollout_id}";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_TRIALS: &str =
    "/v2/runtime/sessions/{session_id}/queue/trials";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_CONTRACT: &str =
    "/v2/runtime/sessions/{session_id}/queue/contract";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_TRIALS_TRIAL_ID: &str =
    "/v2/runtime/sessions/{session_id}/queue/trials/{trial_id}";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_ROLLOUTS: &str =
    "/v2/runtime/sessions/{session_id}/queue/rollouts";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_ROLLOUTS_LEASE: &str =
    "/v2/runtime/sessions/{session_id}/queue/rollouts/lease";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_ROLLOUTS_EXPIRE_LEASES: &str =
    "/v2/runtime/sessions/{session_id}/queue/rollouts/expire-leases";
pub const V2_RUNTIME_SESSIONS_SESSION_ID_QUEUE_ROLLOUTS_ROLLOUT_ID: &str =
    "/v2/runtime/sessions/{session_id}/queue/rollouts/{rollout_id}";
pub const V2_RUNTIME_CONTAINERS_CONTAINER_ID_ROLLOUTS_ROLLOUT_ID_CHECKPOINT_DUMP: &str =
    "/v2/runtime/containers/{container_id}/rollouts/{rollout_id}/checkpoint/dump";
pub const V2_RUNTIME_CONTAINERS_CONTAINER_ID_ROLLOUTS_ROLLOUT_ID_CHECKPOINT_RESTORE: &str =
    "/v2/runtime/containers/{container_id}/rollouts/{rollout_id}/checkpoint/restore";

pub fn v2_online_session(session_id: &str) -> String {
    format!("/v2/online/sessions/{session_id}")
}

pub fn v2_online_session_reward(session_id: &str) -> String {
    format!("/v2/online/sessions/{session_id}/reward")
}

pub fn v2_online_session_events(session_id: &str) -> String {
    format!("/v2/online/sessions/{session_id}/events")
}

pub fn v2_runtime_compatibility() -> &'static str {
    V2_RUNTIME_COMPATIBILITY
}

pub fn v1_optimizer_events() -> &'static str {
    V1_OPTIMIZER_EVENTS
}

pub fn v2_optimizer_events() -> &'static str {
    V2_OPTIMIZER_EVENTS
}

pub fn v1_failures_query() -> &'static str {
    V1_FAILURES_QUERY
}

pub fn v2_failures_query() -> &'static str {
    V2_FAILURES_QUERY
}

pub fn v1_admin_optimizer_events() -> &'static str {
    V1_ADMIN_OPTIMIZER_EVENTS
}

pub fn v2_admin_optimizer_events() -> &'static str {
    V2_ADMIN_OPTIMIZER_EVENTS
}

pub fn v1_admin_failures_query() -> &'static str {
    V1_ADMIN_FAILURES_QUERY
}

pub fn v2_admin_failures_query() -> &'static str {
    V2_ADMIN_FAILURES_QUERY
}

pub fn v1_admin_victoria_logs_query() -> &'static str {
    V1_ADMIN_VICTORIA_LOGS_QUERY
}

pub fn v2_admin_victoria_logs_query() -> &'static str {
    V2_ADMIN_VICTORIA_LOGS_QUERY
}

pub fn v2_runtime_queue_trials(system_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/trials")
}

pub fn v2_runtime_queue_contract(system_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/contract")
}

pub fn v2_runtime_queue_trial(system_id: &str, trial_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/trials/{trial_id}")
}

pub fn v2_runtime_queue_rollouts(system_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/rollouts")
}

pub fn v2_runtime_queue_rollout_lease(system_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/rollouts/lease")
}

pub fn v2_runtime_queue_rollout_expire_leases(system_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/rollouts/expire-leases")
}

pub fn v2_runtime_queue_rollout(system_id: &str, rollout_id: &str) -> String {
    format!("/v2/runtime/systems/{system_id}/queue/rollouts/{rollout_id}")
}

pub fn v2_runtime_session_queue_trials(session_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/trials")
}

pub fn v2_runtime_session_queue_contract(session_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/contract")
}

pub fn v2_runtime_session_queue_trial(session_id: &str, trial_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/trials/{trial_id}")
}

pub fn v2_runtime_session_queue_rollouts(session_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/rollouts")
}

pub fn v2_runtime_session_queue_rollout_lease(session_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/rollouts/lease")
}

pub fn v2_runtime_session_queue_rollout_expire_leases(session_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/rollouts/expire-leases")
}

pub fn v2_runtime_session_queue_rollout(session_id: &str, rollout_id: &str) -> String {
    format!("/v2/runtime/sessions/{session_id}/queue/rollouts/{rollout_id}")
}

pub fn v2_runtime_container_rollout_checkpoint_dump(
    container_id: &str,
    rollout_id: &str,
) -> String {
    format!("/v2/runtime/containers/{container_id}/rollouts/{rollout_id}/checkpoint/dump")
}

pub fn v2_runtime_container_rollout_checkpoint_restore(
    container_id: &str,
    rollout_id: &str,
) -> String {
    format!("/v2/runtime/containers/{container_id}/rollouts/{rollout_id}/checkpoint/restore")
}

pub const V1_SYSTEMS: &str = "/v1/systems";
pub const V1_SYSTEMS_SYSTEM_ID: &str = "/v1/systems/{system_id}";

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
pub const V1_POOLS_POOL_ID_ROLLOUTS_ROLLOUT_ID: &str = "/v1/pools/{pool_id}/rollouts/{rollout_id}";
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

pub fn api_v1_container(container_id: &str) -> String {
    format!("/api/v1/containers/{container_id}")
}

pub fn api_v1_synthtunnel_lease(lease_id: &str) -> String {
    format!("/api/v1/synthtunnel/leases/{lease_id}")
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

pub fn v1_system(system_id: &str) -> String {
    format!("/v1/systems/{system_id}")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_runtime_compatibility_path_is_stable() {
        assert_eq!(V2_RUNTIME_COMPATIBILITY, "/v2/runtime/compatibility");
        assert_eq!(v2_runtime_compatibility(), "/v2/runtime/compatibility");
    }

    #[test]
    fn optimizer_and_failure_query_paths_are_stable() {
        assert_eq!(v1_optimizer_events(), "/v1/optimizer/events");
        assert_eq!(v2_optimizer_events(), "/v2/optimizer/events");
        assert_eq!(v1_failures_query(), "/v1/failures/query");
        assert_eq!(v2_failures_query(), "/v2/failures/query");
        assert_eq!(v1_admin_optimizer_events(), "/v1/admin/optimizer/events");
        assert_eq!(v2_admin_optimizer_events(), "/v2/admin/optimizer/events");
        assert_eq!(v1_admin_failures_query(), "/v1/admin/failures/query");
        assert_eq!(v2_admin_failures_query(), "/v2/admin/failures/query");
        assert_eq!(
            v1_admin_victoria_logs_query(),
            "/v1/admin/victoria-logs/query"
        );
        assert_eq!(
            v2_admin_victoria_logs_query(),
            "/v2/admin/victoria-logs/query"
        );
    }

    #[test]
    fn v2_runtime_queue_paths_are_stable() {
        assert_eq!(
            v2_runtime_queue_trials("sys-1"),
            "/v2/runtime/systems/sys-1/queue/trials"
        );
        assert_eq!(
            v2_runtime_queue_contract("sys-1"),
            "/v2/runtime/systems/sys-1/queue/contract"
        );
        assert_eq!(
            v2_runtime_queue_trial("sys-1", "trial-1"),
            "/v2/runtime/systems/sys-1/queue/trials/trial-1"
        );
        assert_eq!(
            v2_runtime_queue_rollouts("sys-1"),
            "/v2/runtime/systems/sys-1/queue/rollouts"
        );
        assert_eq!(
            v2_runtime_queue_rollout_lease("sys-1"),
            "/v2/runtime/systems/sys-1/queue/rollouts/lease"
        );
        assert_eq!(
            v2_runtime_queue_rollout_expire_leases("sys-1"),
            "/v2/runtime/systems/sys-1/queue/rollouts/expire-leases"
        );
        assert_eq!(
            v2_runtime_queue_rollout("sys-1", "rollout-1"),
            "/v2/runtime/systems/sys-1/queue/rollouts/rollout-1"
        );
        assert_eq!(
            v2_runtime_session_queue_trials("sess-1"),
            "/v2/runtime/sessions/sess-1/queue/trials"
        );
        assert_eq!(
            v2_runtime_session_queue_contract("sess-1"),
            "/v2/runtime/sessions/sess-1/queue/contract"
        );
        assert_eq!(
            v2_runtime_session_queue_trial("sess-1", "trial-1"),
            "/v2/runtime/sessions/sess-1/queue/trials/trial-1"
        );
        assert_eq!(
            v2_runtime_session_queue_rollouts("sess-1"),
            "/v2/runtime/sessions/sess-1/queue/rollouts"
        );
        assert_eq!(
            v2_runtime_session_queue_rollout_lease("sess-1"),
            "/v2/runtime/sessions/sess-1/queue/rollouts/lease"
        );
        assert_eq!(
            v2_runtime_session_queue_rollout_expire_leases("sess-1"),
            "/v2/runtime/sessions/sess-1/queue/rollouts/expire-leases"
        );
        assert_eq!(
            v2_runtime_session_queue_rollout("sess-1", "rollout-1"),
            "/v2/runtime/sessions/sess-1/queue/rollouts/rollout-1"
        );
        assert_eq!(
            v2_runtime_container_rollout_checkpoint_dump("ctr-1", "rollout-1"),
            "/v2/runtime/containers/ctr-1/rollouts/rollout-1/checkpoint/dump"
        );
        assert_eq!(
            v2_runtime_container_rollout_checkpoint_restore("ctr-1", "rollout-1"),
            "/v2/runtime/containers/ctr-1/rollouts/rollout-1/checkpoint/restore"
        );
    }
}
