use std::sync::Arc;

use crate::models::{JsonMap, ResourceList};
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostedContainerSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostedContainer {
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub container_url: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct ContainersClient {
    transport: Arc<Transport>,
}

impl ContainersClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub async fn create(&self, request: &HostedContainerSpec) -> Result<HostedContainer> {
        self.transport
            .post_json(openapi_paths::API_V1_CONTAINERS, request)
            .await
    }

    pub async fn list(&self) -> Result<ResourceList<HostedContainer>> {
        self.transport
            .get_json(openapi_paths::API_V1_CONTAINERS)
            .await
    }

    pub async fn get(&self, container_id: &str) -> Result<HostedContainer> {
        self.transport
            .get_json(&openapi_paths::api_v1_container(container_id))
            .await
    }

    pub async fn delete(&self, container_id: &str) -> Result<()> {
        self.transport
            .delete_empty(&openapi_paths::api_v1_container(container_id))
            .await
    }
}
