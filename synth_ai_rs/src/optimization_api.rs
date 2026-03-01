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

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PolicyOptimizationSystem>> {
        match params {
            Some(query) => {
                self.transport
                    .get_json_with_query(openapi_paths::V1_SYSTEMS, query)
                    .await
            }
            None => {
                self.transport
                    .get_json(openapi_paths::V1_SYSTEMS)
                    .await
            }
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
    pub async fn create(&self, request: &OfflineJobCreateRequest) -> Result<PolicyOptimizationOfflineJob> {
        self.transport
            .post_json(openapi_paths::V1_OFFLINE_JOBS, request)
            .await
    }

    pub async fn get(&self, job_id: &str) -> Result<PolicyOptimizationOfflineJob> {
        self.transport
            .get_json(&openapi_paths::v1_offline_job(job_id))
            .await
    }

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PolicyOptimizationOfflineJob>> {
        match params {
            Some(query) => self
                .transport
                .get_json_with_query(openapi_paths::V1_OFFLINE_JOBS, query)
                .await,
            None => self.transport.get_json(openapi_paths::V1_OFFLINE_JOBS).await,
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

    pub async fn events(&self, job_id: &str, params: Option<&CursorParams>) -> Result<EventsResponse> {
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

    pub async fn cancel(&self, job_id: &str, reason: Option<String>) -> Result<PolicyOptimizationOfflineJob> {
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

    pub async fn pause(&self, job_id: &str, reason: Option<String>) -> Result<PolicyOptimizationOfflineJob> {
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

    pub async fn resume(&self, job_id: &str, reason: Option<String>) -> Result<PolicyOptimizationOfflineJob> {
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
            Some(query) => self
                .transport
                .get_json_with_query(openapi_paths::V1_ONLINE_SESSIONS, query)
                .await,
            None => self.transport.get_json(openapi_paths::V1_ONLINE_SESSIONS).await,
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
            .post_json(&openapi_paths::v1_online_session_reward(session_id), request)
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
}
