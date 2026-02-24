use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::JsonMap;
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<Value>,
    #[serde(default)]
    pub usage: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceJobCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceJob {
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct InferenceClient {
    transport: Arc<Transport>,
}

impl InferenceClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub fn chat(&self) -> InferenceChatClient {
        InferenceChatClient {
            transport: self.transport.clone(),
        }
    }

    pub fn jobs(&self) -> InferenceJobsClient {
        InferenceJobsClient {
            transport: self.transport.clone(),
        }
    }
}

#[derive(Clone)]
pub struct InferenceChatClient {
    transport: Arc<Transport>,
}

impl InferenceChatClient {
    pub fn completions(&self) -> InferenceChatCompletionsClient {
        InferenceChatCompletionsClient {
            transport: self.transport.clone(),
        }
    }
}

#[derive(Clone)]
pub struct InferenceChatCompletionsClient {
    transport: Arc<Transport>,
}

impl InferenceChatCompletionsClient {
    pub async fn create(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        self.transport
            .post_json(openapi_paths::API_INFERENCE_CHAT_COMPLETIONS, request)
            .await
    }
}

#[derive(Clone)]
pub struct InferenceJobsClient {
    transport: Arc<Transport>,
}

impl InferenceJobsClient {
    pub async fn create(&self, request: &InferenceJobCreateRequest) -> Result<InferenceJob> {
        self.transport
            .post_json(openapi_paths::API_INFERENCE_JOBS, request)
            .await
    }

    pub async fn get(&self, job_id: &str) -> Result<InferenceJob> {
        self.transport
            .get_json(&openapi_paths::api_inference_jobs_job(job_id))
            .await
    }

    pub async fn artifact(&self, job_id: &str, artifact_id: &str) -> Result<Vec<u8>> {
        self.transport
            .get_bytes(&openapi_paths::inference_job_artifact(job_id, artifact_id))
            .await
    }
}
