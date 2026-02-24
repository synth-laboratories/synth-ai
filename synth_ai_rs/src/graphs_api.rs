use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::{JsonMap, ResourceList};
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphCompletionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphCompletionResponse {
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub trace: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphResource {
    #[serde(default)]
    pub graph_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct GraphsClient {
    transport: Arc<Transport>,
}

impl GraphsClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub fn completions(&self) -> GraphCompletionsClient {
        GraphCompletionsClient {
            transport: self.transport.clone(),
        }
    }

    pub async fn list_evolved(&self) -> Result<ResourceList<GraphResource>> {
        self.transport
            .get_json(openapi_paths::GRAPH_EVOLVE_GRAPHS)
            .await
    }
}

#[derive(Clone)]
pub struct GraphCompletionsClient {
    transport: Arc<Transport>,
}

impl GraphCompletionsClient {
    pub async fn create(&self, request: &GraphCompletionRequest) -> Result<GraphCompletionResponse> {
        self.transport
            .post_json(openapi_paths::API_GRAPHS_COMPLETIONS, request)
            .await
    }
}
