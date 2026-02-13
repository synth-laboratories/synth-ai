use std::collections::HashSet;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::types::{Result, SynthError};

pub type RolloutHandler =
    Arc<dyn Fn(RolloutRequest) -> BoxFuture<'static, std::result::Result<RolloutResponse, ContainerError>>
        + Send
        + Sync>;

#[derive(Debug, Clone)]
pub struct ContainerError {
    pub status: StatusCode,
    pub message: String,
}

impl ContainerError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDescriptor {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub splits: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_split: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_url: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LimitsInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<i64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task: TaskDescriptor,
    pub dataset: DatasetInfo,
    pub inference: InferenceInfo,
    pub limits: LimitsInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_metadata: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl TaskInfo {
    pub fn minimal(app_id: impl Into<String>, name: impl Into<String>, description: impl Into<String>) -> Self {
        let task = TaskDescriptor {
            id: app_id.into(),
            name: name.into(),
            description: Some(description.into()),
            version: None,
            extra: Map::new(),
        };
        Self {
            task,
            dataset: DatasetInfo::default(),
            inference: InferenceInfo::default(),
            limits: LimitsInfo::default(),
            task_metadata: None,
            extra: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutRequest {
    pub trace_correlation_id: String,
    pub env: Value,
    pub policy: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_done: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synth_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_overrides: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub override_bundle_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutMetrics {
    pub outcome_reward: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_rewards: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome_objectives: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_objectives: Option<Vec<Map<String, Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_objectives: Option<Vec<Map<String, Value>>>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub details: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResponse {
    pub trace_correlation_id: String,
    #[serde(alias = "metrics")]
    pub reward_info: RolloutMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub override_application_results: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone)]
pub struct ContainerConfig {
    pub task_info: TaskInfo,
    pub rollout: RolloutHandler,
    pub require_api_key: bool,
    pub api_keys: Vec<String>,
}

impl ContainerConfig {
    pub fn new<F, Fut>(
        app_id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        handler: F,
    ) -> Self
    where
        F: Fn(RolloutRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<RolloutResponse, ContainerError>> + Send + 'static,
    {
        let rollout: RolloutHandler = Arc::new(move |req| Box::pin(handler(req)));
        let mut api_keys = Vec::new();
        if let Ok(val) = std::env::var("ENVIRONMENT_API_KEY") {
            api_keys.push(val);
        }
        Self {
            task_info: TaskInfo::minimal(app_id, name, description),
            rollout,
            require_api_key: true,
            api_keys,
        }
    }
}

#[derive(Clone)]
pub struct ContainerApp {
    router: Router,
}

pub fn create_container(config: ContainerConfig) -> ContainerApp {
    let state = Arc::new(config);

    let router = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/task_info", get(task_info))
        .route("/info", get(info))
        .route("/rollout", post(rollout))
        .with_state(state);

    ContainerApp { router }
}

impl ContainerApp {
    pub fn router(&self) -> Router {
        self.router.clone()
    }

    pub async fn run(self, addr: SocketAddr) -> Result<()> {
        axum::Server::bind(&addr)
            .serve(self.router.into_make_service())
            .await
            .map_err(|err| SynthError::UnexpectedResponse(err.to_string()))
    }
}

async fn root() -> Response {
    Json(json!({"status": "ok"})).into_response()
}

async fn health(
    State(config): State<Arc<ContainerConfig>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = authorize(&config, &headers) {
        return resp;
    }
    Json(json!({ "healthy": true })).into_response()
}

async fn task_info(
    State(config): State<Arc<ContainerConfig>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = authorize(&config, &headers) {
        return resp;
    }
    Json(config.task_info.clone()).into_response()
}

async fn info(
    State(config): State<Arc<ContainerConfig>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = authorize(&config, &headers) {
        return resp;
    }
    let task = config.task_info.task.clone();
    let version = task.version.clone();
    let service = json!({
        "task": task,
        "version": version,
    });
    let payload = json!({
        "service": service,
        "dataset": config.task_info.dataset,
        "rubrics": null,
        "inference": config.task_info.inference,
        "limits": config.task_info.limits,
    });
    Json(payload).into_response()
}

async fn rollout(
    State(config): State<Arc<ContainerConfig>>,
    headers: HeaderMap,
    Json(request): Json<RolloutRequest>,
) -> impl IntoResponse {
    if let Err(resp) = authorize(&config, &headers) {
        return resp;
    }
    let handler = config.rollout.clone();
    match handler(request).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(err) => (
            err.status,
            Json(json!({ "error": err.message })),
        )
            .into_response(),
    }
}

fn authorize(config: &ContainerConfig, headers: &HeaderMap) -> std::result::Result<(), axum::response::Response> {
    if !config.require_api_key {
        return Ok(());
    }
    let allowed = api_key_set(config);
    if allowed.is_empty() {
        let resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "ENVIRONMENT_API_KEY is not configured" })),
        )
            .into_response();
        return Err(resp);
    }
    let provided = header_keys(headers);
    if provided.iter().any(|key| allowed.contains(key)) {
        return Ok(());
    }
    let resp = (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "API key missing or invalid" })),
    )
        .into_response();
    Err(resp)
}

fn api_key_set(config: &ContainerConfig) -> HashSet<String> {
    let mut set = HashSet::new();
    for key in &config.api_keys {
        if !key.is_empty() {
            set.insert(key.clone());
        }
    }
    if let Ok(aliases) = std::env::var("ENVIRONMENT_API_KEY_ALIASES") {
        for part in aliases.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                set.insert(trimmed.to_string());
            }
        }
    }
    set
}

fn header_keys(headers: &HeaderMap) -> Vec<String> {
    let mut keys = Vec::new();
    for header in ["x-api-key", "x-api-keys", "authorization"] {
        if let Some(value) = headers.get(header) {
            if let Ok(text) = value.to_str() {
                if header == "authorization" && text.to_lowercase().starts_with("bearer ") {
                    keys.extend(split_keys(&text[7..]));
                } else {
                    keys.extend(split_keys(text));
                }
            }
        }
    }
    keys
}

fn split_keys(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
