//! Task app HTTP client.
//!
//! Client for communicating with task apps via the standard API contract.

use super::types::{HealthResponse, InfoResponse, RolloutRequest, RolloutResponse, TaskInfo};
use crate::errors::CoreError;
use crate::http::HttpClient;
use serde_json::Value;

/// Default timeout in seconds for task app requests.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Client for communicating with task apps.
pub struct TaskAppClient {
    client: HttpClient,
    base_url: String,
}

impl TaskAppClient {
    /// Create a new task app client.
    pub fn new(base_url: &str, api_key: Option<&str>) -> Self {
        let key = api_key.unwrap_or("no-auth");
        let client = HttpClient::new(base_url, key, DEFAULT_TIMEOUT_SECS)
            .expect("Failed to create HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Create a client with custom timeout.
    pub fn with_timeout(base_url: &str, api_key: Option<&str>, timeout_secs: u64) -> Self {
        let key = api_key.unwrap_or("no-auth");
        let client =
            HttpClient::new(base_url, key, timeout_secs).expect("Failed to create HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Check task app health.
    pub async fn health(&self) -> Result<HealthResponse, CoreError> {
        let response: Value = self.client.get("/health", None).await?;
        serde_json::from_value(response)
            .map_err(|e| CoreError::Internal(format!("Failed to parse health response: {}", e)))
    }

    /// Check if the task app is healthy.
    pub async fn is_healthy(&self) -> bool {
        self.health().await.map(|r| r.healthy).unwrap_or(false)
    }

    /// Get service info.
    pub async fn info(&self) -> Result<InfoResponse, CoreError> {
        let response: Value = self.client.get("/info", None).await?;
        serde_json::from_value(response)
            .map_err(|e| CoreError::Internal(format!("Failed to parse info response: {}", e)))
    }

    /// Get task info for specific seeds.
    pub async fn task_info(&self, seeds: Option<&[i64]>) -> Result<Vec<TaskInfo>, CoreError> {
        let path = match seeds {
            Some(s) if !s.is_empty() => {
                let query: String = s
                    .iter()
                    .map(|seed| format!("seed={}", seed))
                    .collect::<Vec<_>>()
                    .join("&");
                format!("/task_info?{}", query)
            }
            _ => "/task_info".to_string(),
        };

        let response: Value = self.client.get(&path, None).await?;

        // Response can be a single TaskInfo or array
        if response.is_array() {
            serde_json::from_value(response)
                .map_err(|e| CoreError::Internal(format!("Failed to parse task_info array: {}", e)))
        } else if response.get("taskset").is_some() {
            // Taskset descriptor response (no seeds provided)
            Ok(vec![])
        } else {
            let info: TaskInfo = serde_json::from_value(response)
                .map_err(|e| CoreError::Internal(format!("Failed to parse task_info: {}", e)))?;
            Ok(vec![info])
        }
    }

    /// Get taskset description (no seeds).
    pub async fn taskset_info(&self) -> Result<Value, CoreError> {
        let response: Value = self.client.get("/task_info", None).await?;
        Ok(response)
    }

    /// Execute a rollout.
    pub async fn rollout(&self, request: &RolloutRequest) -> Result<RolloutResponse, CoreError> {
        let body = serde_json::to_value(request).map_err(|e| {
            CoreError::Internal(format!("Failed to serialize rollout request: {}", e))
        })?;

        let response: Value = self.client.post_json("/rollout", &body).await?;

        serde_json::from_value(response)
            .map_err(|e| CoreError::Internal(format!("Failed to parse rollout response: {}", e)))
    }

    /// Signal that the job is done.
    pub async fn done(&self) -> Result<Value, CoreError> {
        let response: Value = self
            .client
            .post_json("/done", &serde_json::json!({}))
            .await?;
        Ok(response)
    }

    /// Raw GET request to any endpoint.
    pub async fn get(&self, path: &str) -> Result<Value, CoreError> {
        let response: Value = self.client.get(path, None).await?;
        Ok(response)
    }

    /// Raw POST request to any endpoint.
    pub async fn post(&self, path: &str, body: &Value) -> Result<Value, CoreError> {
        let response: Value = self.client.post_json(path, body).await?;
        Ok(response)
    }

    /// Wait for the task app to become healthy.
    pub async fn wait_for_healthy(
        &self,
        timeout_seconds: f64,
        poll_interval_seconds: f64,
    ) -> Result<(), CoreError> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_seconds);
        let interval = std::time::Duration::from_secs_f64(poll_interval_seconds);

        loop {
            if start.elapsed() >= timeout {
                return Err(CoreError::Timeout(format!(
                    "Task app at {} did not become healthy within {} seconds",
                    self.base_url, timeout_seconds
                )));
            }

            match self.health().await {
                Ok(health) if health.healthy => return Ok(()),
                Ok(_) | Err(_) => {
                    tokio::time::sleep(interval).await;
                }
            }
        }
    }
}

/// Environment client for RL-style interactions.
pub struct EnvClient<'a> {
    client: &'a TaskAppClient,
}

impl<'a> EnvClient<'a> {
    /// Create a new environment client.
    pub fn new(client: &'a TaskAppClient) -> Self {
        Self { client }
    }

    /// Initialize an environment.
    pub async fn initialize(&self, env_name: &str, payload: &Value) -> Result<Value, CoreError> {
        let path = format!("/env/{}/initialize", env_name);
        self.client.post(&path, payload).await
    }

    /// Take a step in the environment.
    pub async fn step(&self, env_name: &str, payload: &Value) -> Result<Value, CoreError> {
        let path = format!("/env/{}/step", env_name);
        self.client.post(&path, payload).await
    }

    /// Terminate the environment.
    pub async fn terminate(&self, env_name: &str, payload: &Value) -> Result<Value, CoreError> {
        let path = format!("/env/{}/terminate", env_name);
        self.client.post(&path, payload).await
    }

    /// Reset the environment.
    pub async fn reset(&self, env_name: &str, payload: &Value) -> Result<Value, CoreError> {
        let path = format!("/env/{}/reset", env_name);
        self.client.post(&path, payload).await
    }
}

impl TaskAppClient {
    /// Get an environment client for RL-style interactions.
    pub fn env(&self) -> EnvClient<'_> {
        EnvClient::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = TaskAppClient::new("https://task-app.example.com", Some("sk-test"));
        assert_eq!(client.base_url(), "https://task-app.example.com");
    }

    #[test]
    fn test_client_url_normalization() {
        let client = TaskAppClient::new("https://task-app.example.com/", Some("sk-test"));
        assert_eq!(client.base_url(), "https://task-app.example.com");
    }

    #[test]
    fn test_env_client() {
        let client = TaskAppClient::new("https://task-app.example.com", None);
        let _env = client.env();
        // Just verify it compiles
    }
}
