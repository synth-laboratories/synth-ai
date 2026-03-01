use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::{ArtifactsResponse, CursorParams, EventsResponse, JsonMap, ResourceList};
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolUpload {
    #[serde(default)]
    pub upload_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolUploadCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolDataSource {
    #[serde(default)]
    pub data_source_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolDataSourceCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolDataSourceUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolAssembly {
    #[serde(default)]
    pub assembly_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolAssemblyCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Pool {
    #[serde(default)]
    pub pool_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolRollout {
    #[serde(default)]
    pub rollout_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolRolloutCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolTask {
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolTaskCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolTaskUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolMetricsResponse {
    #[serde(default)]
    pub metrics: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct PoolsClient {
    transport: Arc<Transport>,
}

impl PoolsClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub fn uploads(&self) -> PoolUploadsClient {
        PoolUploadsClient {
            transport: self.transport.clone(),
        }
    }

    pub fn data_sources(&self) -> PoolDataSourcesClient {
        PoolDataSourcesClient {
            transport: self.transport.clone(),
        }
    }

    pub fn assemblies(&self) -> PoolAssembliesClient {
        PoolAssembliesClient {
            transport: self.transport.clone(),
        }
    }

    pub async fn create(&self, request: &PoolCreateRequest) -> Result<Pool> {
        self.transport.post_json(openapi_paths::V1_POOLS, request).await
    }

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<Pool>> {
        match params {
            Some(query) => self
                .transport
                .get_json_with_query(openapi_paths::V1_POOLS, query)
                .await,
            None => self.transport.get_json(openapi_paths::V1_POOLS).await,
        }
    }

    pub async fn get(&self, pool_id: &str) -> Result<Pool> {
        self.transport
            .get_json(&openapi_paths::v1_pool(pool_id))
            .await
    }

    pub async fn update(&self, pool_id: &str, request: &PoolUpdateRequest) -> Result<Pool> {
        self.transport
            .patch_json(&openapi_paths::v1_pool(pool_id), request)
            .await
    }

    pub async fn delete(&self, pool_id: &str) -> Result<()> {
        self.transport
            .delete_empty(&openapi_paths::v1_pool(pool_id))
            .await
    }

    pub async fn create_pool_assembly(
        &self,
        pool_id: &str,
        request: &PoolAssemblyCreateRequest,
    ) -> Result<PoolAssembly> {
        self.transport
            .post_json(&openapi_paths::v1_pool_assemblies(pool_id), request)
            .await
    }

    pub fn pool_rollouts(&self, pool_id: impl Into<String>) -> PoolRolloutsClient {
        PoolRolloutsClient {
            transport: self.transport.clone(),
            pool_id: pool_id.into(),
        }
    }

    pub fn pool_tasks(&self, pool_id: impl Into<String>) -> PoolTasksClient {
        PoolTasksClient {
            transport: self.transport.clone(),
            pool_id: pool_id.into(),
        }
    }

    pub async fn metrics(&self, pool_id: &str) -> Result<PoolMetricsResponse> {
        self.transport
            .get_json(&openapi_paths::v1_pool_metrics(pool_id))
            .await
    }
}

#[derive(Clone)]
pub struct PoolUploadsClient {
    transport: Arc<Transport>,
}

impl PoolUploadsClient {
    pub async fn create(&self, request: &PoolUploadCreateRequest) -> Result<PoolUpload> {
        self.transport
            .post_json(openapi_paths::V1_POOLS_UPLOADS, request)
            .await
    }

    pub async fn get(&self, upload_id: &str) -> Result<PoolUpload> {
        self.transport
            .get_json(&openapi_paths::v1_pools_upload(upload_id))
            .await
    }
}

#[derive(Clone)]
pub struct PoolDataSourcesClient {
    transport: Arc<Transport>,
}

impl PoolDataSourcesClient {
    pub async fn create(&self, request: &PoolDataSourceCreateRequest) -> Result<PoolDataSource> {
        self.transport
            .post_json(openapi_paths::V1_POOLS_DATA_SOURCES, request)
            .await
    }

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PoolDataSource>> {
        match params {
            Some(query) => self
                .transport
                .get_json_with_query(openapi_paths::V1_POOLS_DATA_SOURCES, query)
                .await,
            None => self
                .transport
                .get_json(openapi_paths::V1_POOLS_DATA_SOURCES)
                .await,
        }
    }

    pub async fn get(&self, data_source_id: &str) -> Result<PoolDataSource> {
        self.transport
            .get_json(&openapi_paths::v1_pools_data_source(data_source_id))
            .await
    }

    pub async fn update(
        &self,
        data_source_id: &str,
        request: &PoolDataSourceUpdateRequest,
    ) -> Result<PoolDataSource> {
        self.transport
            .patch_json(&openapi_paths::v1_pools_data_source(data_source_id), request)
            .await
    }

    pub async fn refresh(&self, data_source_id: &str) -> Result<PoolDataSource> {
        self.transport
            .post_json(
                &openapi_paths::v1_pools_data_source_refresh(data_source_id),
                &serde_json::json!({}),
            )
            .await
    }
}

#[derive(Clone)]
pub struct PoolAssembliesClient {
    transport: Arc<Transport>,
}

impl PoolAssembliesClient {
    pub async fn create(&self, request: &PoolAssemblyCreateRequest) -> Result<PoolAssembly> {
        self.transport
            .post_json(openapi_paths::V1_POOLS_ASSEMBLIES, request)
            .await
    }

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PoolAssembly>> {
        match params {
            Some(query) => self
                .transport
                .get_json_with_query(openapi_paths::V1_POOLS_ASSEMBLIES, query)
                .await,
            None => self
                .transport
                .get_json(openapi_paths::V1_POOLS_ASSEMBLIES)
                .await,
        }
    }

    pub async fn get(&self, assembly_id: &str) -> Result<PoolAssembly> {
        self.transport
            .get_json(&openapi_paths::v1_pools_assembly(assembly_id))
            .await
    }

    pub async fn events(&self, assembly_id: &str, params: Option<&CursorParams>) -> Result<EventsResponse> {
        let path = openapi_paths::v1_pools_assembly_events(assembly_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }
}

#[derive(Clone)]
pub struct PoolRolloutsClient {
    transport: Arc<Transport>,
    pool_id: String,
}

impl PoolRolloutsClient {
    pub async fn create(&self, request: &PoolRolloutCreateRequest) -> Result<PoolRollout> {
        self.transport
            .post_json(&openapi_paths::v1_pool_rollouts(&self.pool_id), request)
            .await
    }

    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PoolRollout>> {
        let path = openapi_paths::v1_pool_rollouts(&self.pool_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }

    pub async fn get(&self, rollout_id: &str) -> Result<PoolRollout> {
        self.transport
            .get_json(&openapi_paths::v1_pool_rollout(&self.pool_id, rollout_id))
            .await
    }

    pub async fn cancel(&self, rollout_id: &str) -> Result<PoolRollout> {
        self.transport
            .post_json(
                &openapi_paths::v1_pool_rollout_cancel(&self.pool_id, rollout_id),
                &serde_json::json!({}),
            )
            .await
    }

    pub async fn artifacts(&self, rollout_id: &str) -> Result<ArtifactsResponse> {
        self.transport
            .get_json(&openapi_paths::v1_pool_rollout_artifacts(&self.pool_id, rollout_id))
            .await
    }

    pub async fn usage(&self, rollout_id: &str) -> Result<Value> {
        self.transport
            .get_json(&openapi_paths::v1_pool_rollout_usage(&self.pool_id, rollout_id))
            .await
    }

    pub async fn events(&self, rollout_id: &str, params: Option<&CursorParams>) -> Result<EventsResponse> {
        let path = openapi_paths::v1_pool_rollout_events(&self.pool_id, rollout_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }
}

#[derive(Clone)]
pub struct PoolTasksClient {
    transport: Arc<Transport>,
    pool_id: String,
}

impl PoolTasksClient {
    pub async fn list(&self, params: Option<&CursorParams>) -> Result<ResourceList<PoolTask>> {
        let path = openapi_paths::v1_pool_tasks(&self.pool_id);
        match params {
            Some(query) => self.transport.get_json_with_query(&path, query).await,
            None => self.transport.get_json(&path).await,
        }
    }

    pub async fn create(&self, request: &PoolTaskCreateRequest) -> Result<PoolTask> {
        self.transport
            .post_json(&openapi_paths::v1_pool_tasks(&self.pool_id), request)
            .await
    }

    pub async fn update(&self, task_id: &str, request: &PoolTaskUpdateRequest) -> Result<PoolTask> {
        self.transport
            .put_json(&openapi_paths::v1_pool_task(&self.pool_id, task_id), request)
            .await
    }

    pub async fn delete(&self, task_id: &str) -> Result<()> {
        self.transport
            .delete_empty(&openapi_paths::v1_pool_task(&self.pool_id, task_id))
            .await
    }
}

