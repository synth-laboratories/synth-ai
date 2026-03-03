use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::{ArtifactsResponse, CursorParams, EventsResponse, JsonMap, ResourceList};
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyOptimizationSystem {
    #[serde(default)]
    pub system_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyOptimizationOfflineJob {
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyOptimizationOnlineSession {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemCreateRequest {
    #[serde(flatten)]
    pub body: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemUpdateRequest {
    #[serde(flatten)]
    pub body: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OfflineJobCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OfflineJobUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnlineSessionCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnlineSessionUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueueQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_state_revision: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerEventsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollout_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_tick_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_redacted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminVictoriaLogsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueueCreateTrialRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_id: Option<String>,
    pub candidate_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonMap>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueuePatchTrialRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueueEnqueueRolloutRequest {
    pub trial_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub not_before_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueueLeaseRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub now_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueuePatchRolloutRequest {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub now_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeQueuePatchContractRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_contract: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_override: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnlineRewardRequest {
    pub trace_correlation_id: String,
    pub reward_info: RewardInfo,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RewardInfo {
    pub outcome_reward: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_rewards: Option<Vec<f64>>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct OptimizationClient {
    transport: Arc<Transport>,
}

impl OptimizationClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub fn systems(&self) -> SystemsClient {
        SystemsClient {
            transport: self.transport.clone(),
        }
    }

    pub fn offline(&self) -> OfflineJobsClient {
        OfflineJobsClient {
            transport: self.transport.clone(),
        }
    }

    pub fn online(&self) -> OnlineSessionsClient {
        OnlineSessionsClient {
            transport: self.transport.clone(),
        }
    }
}

#[derive(Clone)]
pub struct SystemsClient {
    transport: Arc<Transport>,
}

impl SystemsClient {
    pub async fn create(&self, request: &SystemCreateRequest) -> Result<PolicyOptimizationSystem> {
        self.transport
            .post_json(openapi_paths::V1_SYSTEMS, request)
            .await
    }

    pub async fn get(&self, system_id: &str) -> Result<PolicyOptimizationSystem> {
        self.transport
            .get_json(&openapi_paths::v1_system(system_id))
            .await
    }

    pub async fn list(
        &self,
        params: Option<&CursorParams>,
    ) -> Result<ResourceList<PolicyOptimizationSystem>> {
        match params {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_SYSTEMS, query)
                    .await
            }
            None => self.transport.get_json(openapi_paths::V1_SYSTEMS).await,
        }
    }

    pub async fn update(
        &self,
        system_id: &str,
        request: &SystemUpdateRequest,
    ) -> Result<PolicyOptimizationSystem> {
        self.transport
            .patch_json(&openapi_paths::v1_system(system_id), request)
            .await
    }

    pub async fn delete(&self, system_id: &str) -> Result<()> {
        self.transport
            .delete_empty(&openapi_paths::v1_system(system_id))
            .await
    }
}

#[derive(Clone)]
pub struct OfflineJobsClient {
    transport: Arc<Transport>,
}

impl OfflineJobsClient {
    pub async fn create(
        &self,
        request: &OfflineJobCreateRequest,
    ) -> Result<PolicyOptimizationOfflineJob> {
        self.transport
            .post_json(openapi_paths::V1_OFFLINE_JOBS, request)
            .await
    }

    pub async fn get(&self, job_id: &str) -> Result<PolicyOptimizationOfflineJob> {
        self.transport
            .get_json(&openapi_paths::v1_offline_job(job_id))
            .await
    }

    pub async fn list(
        &self,
        params: Option<&CursorParams>,
    ) -> Result<ResourceList<PolicyOptimizationOfflineJob>> {
        match params {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_OFFLINE_JOBS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_OFFLINE_JOBS)
                    .await
            }
        }
    }

    pub async fn update(
        &self,
        job_id: &str,
        request: &OfflineJobUpdateRequest,
    ) -> Result<PolicyOptimizationOfflineJob> {
        self.transport
            .patch_json(&openapi_paths::v1_offline_job(job_id), request)
            .await
    }

    pub async fn events(
        &self,
        job_id: &str,
        params: Option<&CursorParams>,
    ) -> Result<EventsResponse> {
        let path = openapi_paths::v1_offline_job_events(job_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }

    pub async fn artifacts(&self, job_id: &str) -> Result<ArtifactsResponse> {
        self.transport
            .get_json(&openapi_paths::v1_offline_job_artifacts(job_id))
            .await
    }

    pub async fn cancel(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOfflineJob> {
        self.update(
            job_id,
            &OfflineJobUpdateRequest {
                state: Some("cancelled".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }

    pub async fn pause(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOfflineJob> {
        self.update(
            job_id,
            &OfflineJobUpdateRequest {
                state: Some("paused".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }

    pub async fn resume(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOfflineJob> {
        self.update(
            job_id,
            &OfflineJobUpdateRequest {
                state: Some("running".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }
}

#[derive(Clone)]
pub struct OnlineSessionsClient {
    transport: Arc<Transport>,
}

fn append_runtime_queue_query(path: String, query: Option<&RuntimeQueueQuery>) -> String {
    let Some(query) = query else {
        return path;
    };
    let mut items: Vec<String> = Vec::new();
    if let Some(actor) = &query.actor {
        items.push(format!("actor={actor}"));
    }
    if let Some(algorithm) = &query.algorithm {
        items.push(format!("algorithm={algorithm}"));
    }
    if let Some(status) = &query.status {
        items.push(format!("status={status}"));
    }
    if let Some(limit) = query.limit {
        items.push(format!("limit={limit}"));
    }
    if let Some(expected_state_revision) = query.expected_state_revision {
        items.push(format!("expected_state_revision={expected_state_revision}"));
    }
    if items.is_empty() {
        return path;
    }
    format!("{path}?{}", items.join("&"))
}

impl OnlineSessionsClient {
    pub async fn create(
        &self,
        request: &OnlineSessionCreateRequest,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.transport
            .post_json(openapi_paths::V1_ONLINE_SESSIONS, request)
            .await
    }

    pub async fn get(&self, session_id: &str) -> Result<PolicyOptimizationOnlineSession> {
        self.transport
            .get_json(&openapi_paths::v1_online_session(session_id))
            .await
    }

    pub async fn list(
        &self,
        params: Option<&CursorParams>,
    ) -> Result<ResourceList<PolicyOptimizationOnlineSession>> {
        match params {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_ONLINE_SESSIONS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_ONLINE_SESSIONS)
                    .await
            }
        }
    }

    pub async fn update(
        &self,
        session_id: &str,
        request: &OnlineSessionUpdateRequest,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.transport
            .patch_json(&openapi_paths::v1_online_session(session_id), request)
            .await
    }

    pub async fn reward(
        &self,
        session_id: &str,
        request: &OnlineRewardRequest,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.transport
            .post_json(
                &openapi_paths::v1_online_session_reward(session_id),
                request,
            )
            .await
    }

    pub async fn events(
        &self,
        session_id: &str,
        params: Option<&CursorParams>,
    ) -> Result<EventsResponse> {
        let path = openapi_paths::v1_online_session_events(session_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }

    pub async fn cancel(
        &self,
        session_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.update(
            session_id,
            &OnlineSessionUpdateRequest {
                state: Some("cancelled".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }

    pub async fn pause(
        &self,
        session_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.update(
            session_id,
            &OnlineSessionUpdateRequest {
                state: Some("paused".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }

    pub async fn resume(
        &self,
        session_id: &str,
        reason: Option<String>,
    ) -> Result<PolicyOptimizationOnlineSession> {
        self.update(
            session_id,
            &OnlineSessionUpdateRequest {
                state: Some("running".to_string()),
                reason,
                extra: JsonMap::new(),
            },
        )
        .await
    }

    /// Fetch the v2 runtime route compatibility contract.
    pub async fn runtime_compatibility(&self) -> Result<Value> {
        self.transport
            .get_json(openapi_paths::V2_RUNTIME_COMPATIBILITY)
            .await
    }

    pub async fn runtime_container_rollout_checkpoint_dump(
        &self,
        container_id: &str,
        rollout_id: &str,
        request: Option<&Value>,
    ) -> Result<Value> {
        let path =
            openapi_paths::v2_runtime_container_rollout_checkpoint_dump(container_id, rollout_id);
        let payload = request
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        self.transport.post_json(&path, &payload).await
    }

    pub async fn runtime_container_rollout_checkpoint_restore(
        &self,
        container_id: &str,
        rollout_id: &str,
        request: Option<&Value>,
    ) -> Result<Value> {
        let path = openapi_paths::v2_runtime_container_rollout_checkpoint_restore(
            container_id,
            rollout_id,
        );
        let payload = request
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        self.transport.post_json(&path, &payload).await
    }

    pub async fn optimizer_events_v1(&self, query: Option<&OptimizerEventsQuery>) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_OPTIMIZER_EVENTS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_OPTIMIZER_EVENTS)
                    .await
            }
        }
    }

    pub async fn optimizer_events_v2(&self, query: Option<&OptimizerEventsQuery>) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V2_OPTIMIZER_EVENTS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V2_OPTIMIZER_EVENTS)
                    .await
            }
        }
    }

    pub async fn failure_events_v1(&self, query: Option<&OptimizerEventsQuery>) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_FAILURES_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_FAILURES_QUERY)
                    .await
            }
        }
    }

    pub async fn failure_events_v2(&self, query: Option<&OptimizerEventsQuery>) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V2_FAILURES_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V2_FAILURES_QUERY)
                    .await
            }
        }
    }

    pub async fn admin_optimizer_events_v1(
        &self,
        query: Option<&OptimizerEventsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_ADMIN_OPTIMIZER_EVENTS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_ADMIN_OPTIMIZER_EVENTS)
                    .await
            }
        }
    }

    pub async fn admin_optimizer_events_v2(
        &self,
        query: Option<&OptimizerEventsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V2_ADMIN_OPTIMIZER_EVENTS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V2_ADMIN_OPTIMIZER_EVENTS)
                    .await
            }
        }
    }

    pub async fn admin_failure_events_v1(
        &self,
        query: Option<&OptimizerEventsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_ADMIN_FAILURES_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_ADMIN_FAILURES_QUERY)
                    .await
            }
        }
    }

    pub async fn admin_failure_events_v2(
        &self,
        query: Option<&OptimizerEventsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V2_ADMIN_FAILURES_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V2_ADMIN_FAILURES_QUERY)
                    .await
            }
        }
    }

    pub async fn admin_victoria_logs_query_v1(
        &self,
        query: Option<&AdminVictoriaLogsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_ADMIN_VICTORIA_LOGS_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_ADMIN_VICTORIA_LOGS_QUERY)
                    .await
            }
        }
    }

    pub async fn admin_victoria_logs_query_v2(
        &self,
        query: Option<&AdminVictoriaLogsQuery>,
    ) -> Result<Value> {
        match query {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V2_ADMIN_VICTORIA_LOGS_QUERY, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V2_ADMIN_VICTORIA_LOGS_QUERY)
                    .await
            }
        }
    }

    pub async fn runtime_queue_trials(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_trials(system_id), query);
        self.transport.get_json(&path).await
    }

    pub async fn runtime_queue_contract(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_contract(system_id), query);
        self.transport.get_json(&path).await
    }

    pub async fn runtime_queue_patch_contract(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchContractRequest,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_contract(system_id), query);
        self.transport.patch_json(&path, request).await
    }

    pub async fn runtime_queue_trial(
        &self,
        system_id: &str,
        trial_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_queue_trial(system_id, trial_id),
            query,
        );
        self.transport.get_json(&path).await
    }

    pub async fn runtime_queue_create_trial(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueCreateTrialRequest,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_trials(system_id), query);
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_queue_patch_trial(
        &self,
        system_id: &str,
        trial_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchTrialRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_queue_trial(system_id, trial_id),
            query,
        );
        self.transport.patch_json(&path, request).await
    }

    pub async fn runtime_queue_rollouts(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_rollouts(system_id), query);
        self.transport.get_json(&path).await
    }

    pub async fn runtime_queue_enqueue_rollout(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueEnqueueRolloutRequest,
    ) -> Result<Value> {
        let path =
            append_runtime_queue_query(openapi_paths::v2_runtime_queue_rollouts(system_id), query);
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_queue_lease_rollout(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueLeaseRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_queue_rollout_lease(system_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_queue_expire_rollout_leases(
        &self,
        system_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueLeaseRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_queue_rollout_expire_leases(system_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_queue_patch_rollout(
        &self,
        system_id: &str,
        rollout_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchRolloutRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_queue_rollout(system_id, rollout_id),
            query,
        );
        self.transport.patch_json(&path, request).await
    }

    pub async fn runtime_session_queue_trials(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_trials(session_id),
            query,
        );
        self.transport.get_json(&path).await
    }

    pub async fn runtime_session_queue_contract(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_contract(session_id),
            query,
        );
        self.transport.get_json(&path).await
    }

    pub async fn runtime_session_queue_patch_contract(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchContractRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_contract(session_id),
            query,
        );
        self.transport.patch_json(&path, request).await
    }

    pub async fn runtime_session_queue_trial(
        &self,
        session_id: &str,
        trial_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_trial(session_id, trial_id),
            query,
        );
        self.transport.get_json(&path).await
    }

    pub async fn runtime_session_queue_create_trial(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueCreateTrialRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_trials(session_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_session_queue_patch_trial(
        &self,
        session_id: &str,
        trial_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchTrialRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_trial(session_id, trial_id),
            query,
        );
        self.transport.patch_json(&path, request).await
    }

    pub async fn runtime_session_queue_rollouts(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_rollouts(session_id),
            query,
        );
        self.transport.get_json(&path).await
    }

    pub async fn runtime_session_queue_enqueue_rollout(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueEnqueueRolloutRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_rollouts(session_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_session_queue_lease_rollout(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueLeaseRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_rollout_lease(session_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_session_queue_expire_rollout_leases(
        &self,
        session_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueueLeaseRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_rollout_expire_leases(session_id),
            query,
        );
        self.transport.post_json(&path, request).await
    }

    pub async fn runtime_session_queue_patch_rollout(
        &self,
        session_id: &str,
        rollout_id: &str,
        query: Option<&RuntimeQueueQuery>,
        request: &RuntimeQueuePatchRolloutRequest,
    ) -> Result<Value> {
        let path = append_runtime_queue_query(
            openapi_paths::v2_runtime_session_queue_rollout(session_id, rollout_id),
            query,
        );
        self.transport.patch_json(&path, request).await
    }
}
