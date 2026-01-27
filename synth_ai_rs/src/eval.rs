use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::client::{AuthStyle, SynthClient};
use crate::sse::{stream_sse, SseStream};
use crate::types::{Result, SynthError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalJobConfig {
    pub task_app_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_app_api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub policy: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_config: Option<Value>,
    #[serde(rename = "max_concurrent", skip_serializing_if = "Option::is_none")]
    pub max_concurrent: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

impl EvalJobConfig {
    pub fn new(task_app_url: impl Into<String>) -> Self {
        Self {
            task_app_url: task_app_url.into(),
            task_app_api_key: None,
            app_id: None,
            env_name: None,
            seeds: Vec::new(),
            policy: json!({}),
            env_config: None,
            verifier_config: None,
            max_concurrent: None,
            timeout: None,
        }
    }
}

#[derive(Clone)]
pub struct EvalJob {
    client: SynthClient,
    job_id: String,
}

impl EvalJob {
    pub fn new(client: SynthClient, job_id: impl Into<String>) -> Self {
        Self {
            client,
            job_id: job_id.into(),
        }
    }

    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    pub async fn submit(client: SynthClient, config: &EvalJobConfig) -> Result<Self> {
        let resp = client
            .post_json("/eval/jobs", config, AuthStyle::Both)
            .await?;
        let job_id = resp
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SynthError::UnexpectedResponse("missing job_id".to_string()))?;
        Ok(Self::new(client, job_id))
    }

    pub async fn status(&self) -> Result<Value> {
        let path = format!("/eval/jobs/{}", self.job_id);
        self.client.get_json(&path, AuthStyle::Both).await
    }

    pub async fn results(&self) -> Result<Value> {
        let path = format!("/eval/jobs/{}/results", self.job_id);
        self.client.get_json(&path, AuthStyle::Both).await
    }

    pub async fn stream_events(&self) -> Result<SseStream> {
        let path = format!("/eval/jobs/{}/events/stream", self.job_id);
        let url = self.client.api_base() + &path;
        let headers = self.client.auth_headers(AuthStyle::Both);
        stream_sse(self.client.http(), url, headers).await
    }
}
