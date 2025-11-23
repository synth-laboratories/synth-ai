//! Synth Task App Example - Rust Implementation
//!
//! This is a minimal but complete Task App that implements the Synth contract
//! for prompt optimization. It demonstrates:
//!
//! - `/health` endpoint (unauthenticated)
//! - `/rollout` endpoint (authenticated)
//! - Dataset loading and sample retrieval
//! - Prompt template rendering with placeholders
//! - LLM calls via inference_url
//! - Reward computation
//!
//! ## Running
//!
//! ```bash
//! cargo run --release
//! ```
//!
//! ## Connecting to Optimizer
//!
//! ```bash
//! # Expose via Cloudflare tunnel
//! cloudflared tunnel --url http://localhost:8001
//!
//! # Start optimization (no Python needed)
//! curl -X POST https://api.usesynth.ai/api/prompt-learning/online/jobs \
//!   -H "Authorization: Bearer $SYNTH_API_KEY" \
//!   -H "Content-Type: application/json" \
//!   -d '{
//!     "algorithm": "mipro",
//!     "config_body": {
//!       "prompt_learning": {
//!         "task_app_url": "https://your-tunnel.trycloudflare.com",
//!         "task_app_api_key": "your-env-key"
//!       }
//!     }
//!   }'
//! ```

use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, sync::Arc};
use tracing::{info, warn};

// =============================================================================
// Dataset
// =============================================================================

/// A single sample in our dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sample {
    text: String,
    label: String,
}

/// Simple in-memory dataset
struct Dataset {
    samples: Vec<Sample>,
    labels: Vec<String>,
}

impl Dataset {
    fn new() -> Self {
        // Embedded Banking77-style samples for demonstration
        // In production, load from file or database
        let samples = vec![
            Sample { text: "How do I reset my PIN?".into(), label: "change_pin".into() },
            Sample { text: "My card hasn't arrived yet".into(), label: "card_arrival".into() },
            Sample { text: "I want to cancel my card".into(), label: "card_cancellation".into() },
            Sample { text: "How do I activate my new card?".into(), label: "activate_my_card".into() },
            Sample { text: "What's my current balance?".into(), label: "balance".into() },
            Sample { text: "I need to dispute a transaction".into(), label: "dispute_charge".into() },
            Sample { text: "Can I get a refund?".into(), label: "refund".into() },
            Sample { text: "How do I transfer money?".into(), label: "transfer".into() },
            Sample { text: "I lost my card".into(), label: "lost_card".into() },
            Sample { text: "Is there a fee for this?".into(), label: "fee".into() },
        ];

        let labels: Vec<String> = samples.iter().map(|s| s.label.clone()).collect();
        let mut unique_labels: Vec<String> = labels.clone();
        unique_labels.sort();
        unique_labels.dedup();

        Dataset { samples, labels: unique_labels }
    }

    fn get(&self, index: usize) -> &Sample {
        &self.samples[index % self.samples.len()]
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

// =============================================================================
// Request/Response Types (matching OpenAPI contract)
// =============================================================================

#[derive(Debug, Deserialize)]
struct RolloutRequest {
    run_id: String,
    env: EnvSpec,
    policy: PolicySpec,
    #[serde(default)]
    mode: String,
}

#[derive(Debug, Deserialize)]
struct EnvSpec {
    seed: Option<i64>,
    #[serde(default)]
    config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct PolicySpec {
    policy_id: Option<String>,
    policy_name: Option<String>,
    #[serde(default)]
    config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct RolloutResponse {
    run_id: String,
    trajectories: Vec<Trajectory>,
    metrics: Metrics,
    aborted: bool,
    ops_executed: i32,
}

#[derive(Debug, Serialize)]
struct Trajectory {
    env_id: String,
    policy_id: String,
    steps: Vec<Step>,
    length: i32,
    inference_url: String,
}

#[derive(Debug, Serialize)]
struct Step {
    obs: HashMap<String, serde_json::Value>,
    tool_calls: Vec<ToolCall>,
    reward: f64,
    done: bool,
    info: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCall,
}

#[derive(Debug, Serialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct Metrics {
    episode_returns: Vec<f64>,
    mean_return: f64,
    num_steps: i32,
    num_episodes: i32,
    outcome_score: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    healthy: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    detail: String,
}

// TaskInfo types
#[derive(Debug, Serialize)]
struct TaskInfo {
    task: TaskDescriptor,
    environment: String,
    dataset: DatasetInfo,
    rubric: RubricInfo,
    inference: InferenceInfo,
    limits: LimitsInfo,
}

#[derive(Debug, Serialize)]
struct TaskDescriptor {
    task_id: String,
    name: String,
    description: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    seeds: Vec<i32>,
    train_count: i32,
    val_count: i32,
    test_count: i32,
}

#[derive(Debug, Serialize)]
struct RubricInfo {
    scoring_criteria: String,
    metric_primary: String,
    metric_range: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct InferenceInfo {
    mode: String,
    supported_tools: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LimitsInfo {
    max_response_tokens: i32,
    timeout_seconds: i32,
}

// =============================================================================
// LLM Client
// =============================================================================

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    tool_choice: String,
    temperature: f64,
    max_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

#[derive(Debug, Serialize)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

#[derive(Debug, Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

async fn call_llm(
    client: &reqwest::Client,
    inference_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    api_key: Option<&str>,
    llm_api_key: Option<&str>,
) -> Result<ChatResponse> {
    // Build the full URL - handle query params correctly
    // inference_url may be "http://host/path?query" - we need "http://host/path/chat/completions?query"
    let url = if let Some(query_start) = inference_url.find('?') {
        let (base, query) = inference_url.split_at(query_start);
        format!("{}/chat/completions{}", base.trim_end_matches('/'), query)
    } else {
        format!("{}/chat/completions", inference_url.trim_end_matches('/'))
    };

    info!("LLM call: inference_url={} full_url={} model={}", inference_url, url, model);

    // Define our classification tool
    let tool = Tool {
        tool_type: "function".into(),
        function: ToolFunction {
            name: "classify".into(),
            description: "Classify the customer query into an intent category".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "The classified intent"
                    }
                },
                "required": ["intent"]
            }),
        },
    };

    let request = ChatRequest {
        model: model.into(),
        messages,
        tools: vec![tool],
        tool_choice: "required".into(),
        temperature: 0.0,
        max_tokens: 100,
    };

    let mut req = client.post(&url).json(&request);

    // Forward API key if provided
    if let Some(key) = api_key {
        req = req.header("X-API-Key", key);
    }

    // Add Bearer auth for OpenAI-compatible APIs
    if let Some(key) = llm_api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = req.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("LLM request failed: {} - {}", status, body);
    }

    Ok(response.json().await?)
}

// =============================================================================
// Prompt Rendering
// =============================================================================

fn render_template(template: &str, placeholders: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in placeholders {
        result = result.replace(&format!("{{{}}}", key), value);
    }
    result
}

fn build_messages(
    policy_config: &HashMap<String, serde_json::Value>,
    sample: &Sample,
    labels: &[String],
) -> Vec<ChatMessage> {
    // Check for prompt_template in policy config
    let prompt_template = policy_config.get("prompt_template");

    // Build placeholders
    let mut placeholders = HashMap::new();
    placeholders.insert("query".into(), sample.text.clone());
    placeholders.insert("intents".into(), labels.join(", "));

    if let Some(template) = prompt_template {
        // Use prompt template from optimizer
        let sections = template
            .get("prompt_sections")
            .or_else(|| template.get("sections"))
            .and_then(|s| s.as_array());

        if let Some(sections) = sections {
            let mut messages = Vec::new();
            let mut sorted_sections: Vec<_> = sections.iter().collect();
            sorted_sections.sort_by_key(|s| s.get("order").and_then(|o| o.as_i64()).unwrap_or(0));

            for section in sorted_sections {
                let role = section.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                let content = section
                    .get("content")
                    .or_else(|| section.get("pattern"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                let rendered = render_template(content, &placeholders);
                messages.push(ChatMessage {
                    role: role.into(),
                    content: rendered,
                });
            }
            return messages;
        }
    }

    // Default messages if no template provided
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You are an expert banking assistant that classifies customer queries. \
                      Use the classify tool to return the intent.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!(
                "Customer Query: {}\n\nAvailable intents: {}\n\nClassify this query.",
                sample.text,
                labels.join(", ")
            ),
        },
    ]
}

// =============================================================================
// App State
// =============================================================================

struct AppState {
    dataset: Dataset,
    http_client: reqwest::Client,
    api_key: Option<String>,
    llm_api_key: Option<String>,
}

// =============================================================================
// Handlers
// =============================================================================

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { healthy: true })
}

async fn task_info_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<Vec<TaskInfo>>, (StatusCode, Json<ErrorResponse>)> {
    // Parse seeds from query string manually
    // Handles: ?seed=0&seed=1 OR ?seeds=0&seeds=1 (httpx serializes list params with repeated keys)
    let query_string = req.uri().query().unwrap_or("");
    let seeds: Vec<i32> = query_string
        .split('&')
        .filter_map(|param| {
            let mut parts = param.split('=');
            match (parts.next(), parts.next()) {
                // Handle both singular "seed" and plural "seeds" parameters
                (Some("seed"), Some(val)) | (Some("seeds"), Some(val)) => val.parse().ok(),
                _ => None,
            }
        })
        .collect();

    // Verify API key (same as rollout)
    let provided_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());

    if let Some(expected_key) = &state.api_key {
        match provided_key {
            Some(key) if key == expected_key => {}
            _ => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        detail: "Invalid or missing API key".into(),
                    }),
                ));
            }
        }
    }

    let dataset_size = state.dataset.len() as i32;
    let all_seeds: Vec<i32> = (0..dataset_size).collect();

    // If seeds are specified, return one TaskInfo per seed
    // Otherwise return a single TaskInfo with all seeds
    let seeds_to_return = if seeds.is_empty() {
        vec![all_seeds.clone()]
    } else {
        seeds.iter().map(|s| vec![*s]).collect()
    };

    let infos: Vec<TaskInfo> = seeds_to_return
        .iter()
        .map(|seeds| TaskInfo {
            task: TaskDescriptor {
                task_id: "banking77-rust".into(),
                name: "Banking77 Intent Classification (Rust)".into(),
                description: "Classify banking customer queries into intent categories".into(),
                version: "1.0.0".into(),
            },
            environment: "banking77".into(),
            dataset: DatasetInfo {
                seeds: seeds.clone(),
                train_count: dataset_size,
                val_count: 0,
                test_count: 0,
            },
            rubric: RubricInfo {
                scoring_criteria: "exact_match".into(),
                metric_primary: "accuracy".into(),
                metric_range: vec![0.0, 1.0],
            },
            inference: InferenceInfo {
                mode: "tool_call".into(),
                supported_tools: vec!["classify".into()],
            },
            limits: LimitsInfo {
                max_response_tokens: 100,
                timeout_seconds: 30,
            },
        })
        .collect();

    Ok(Json(infos))
}

async fn rollout_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<RolloutRequest>,
) -> Result<Json<RolloutResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Verify API key
    let provided_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());

    if let Some(expected_key) = &state.api_key {
        match provided_key {
            Some(key) if key == expected_key => {}
            _ => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        detail: "Invalid or missing API key".into(),
                    }),
                ));
            }
        }
    }

    // Extract seed and get sample
    let seed = request.env.seed.unwrap_or(0) as usize;
    let sample = state.dataset.get(seed);

    info!(
        "Rollout: run_id={} seed={} query={}",
        request.run_id, seed, sample.text
    );

    // Get inference URL from policy config
    let inference_url = request
        .policy
        .config
        .get("inference_url")
        .or_else(|| request.policy.config.get("api_base"))
        .or_else(|| request.policy.config.get("base_url"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    detail: "Missing inference_url in policy.config".into(),
                }),
            )
        })?;

    let model = request
        .policy
        .config
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o-mini");

    // Build messages
    let messages = build_messages(&request.policy.config, sample, &state.dataset.labels);

    // Call LLM
    let llm_response = call_llm(
        &state.http_client,
        inference_url,
        model,
        messages,
        provided_key,
        state.llm_api_key.as_deref(),
    )
    .await
    .map_err(|e| {
        warn!("LLM call failed: {}", e);
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                detail: format!("LLM call failed: {}", e),
            }),
        )
    })?;

    // Extract prediction from response
    let (predicted_intent, tool_calls) = extract_prediction(&llm_response);

    // Compute reward
    let is_correct = predicted_intent
        .as_ref()
        .map(|p| p.to_lowercase() == sample.label.to_lowercase())
        .unwrap_or(false);
    let reward = if is_correct { 1.0 } else { 0.0 };

    info!(
        "Result: expected={} predicted={:?} correct={} reward={}",
        sample.label, predicted_intent, is_correct, reward
    );

    // Build response
    let mut obs = HashMap::new();
    obs.insert("query".into(), serde_json::json!(sample.text));
    obs.insert("index".into(), serde_json::json!(seed));

    let mut info = HashMap::new();
    info.insert("expected".into(), serde_json::json!(sample.label));
    info.insert("predicted".into(), serde_json::json!(predicted_intent));
    info.insert("correct".into(), serde_json::json!(is_correct));

    let step = Step {
        obs,
        tool_calls,
        reward,
        done: true,
        info,
    };

    let trajectory = Trajectory {
        env_id: format!("task::train::{}", seed),
        policy_id: request
            .policy
            .policy_id
            .or(request.policy.policy_name)
            .unwrap_or_else(|| "policy".into()),
        steps: vec![step],
        length: 1,
        inference_url: inference_url.into(),
    };

    let response = RolloutResponse {
        run_id: request.run_id,
        trajectories: vec![trajectory],
        metrics: Metrics {
            episode_returns: vec![reward],
            mean_return: reward,
            num_steps: 1,
            num_episodes: 1,
            outcome_score: reward,
        },
        aborted: false,
        ops_executed: 1,
    };

    Ok(Json(response))
}

fn extract_prediction(response: &ChatResponse) -> (Option<String>, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut predicted = None;

    if let Some(choice) = response.choices.first() {
        if let Some(calls) = &choice.message.tool_calls {
            for call in calls {
                if call.function.name == "classify" {
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                        predicted = args.get("intent").and_then(|i| i.as_str()).map(String::from);
                    }
                }

                tool_calls.push(ToolCall {
                    id: call.id.clone(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    },
                });
            }
        }

        // Fallback to content if no tool calls
        if predicted.is_none() {
            if let Some(content) = &choice.message.content {
                predicted = Some(content.trim().to_string());
            }
        }
    }

    (predicted, tool_calls)
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load configuration from environment
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001);

    let api_key = env::var("ENVIRONMENT_API_KEY").ok();

    if api_key.is_some() {
        info!("API key authentication enabled");
    } else {
        warn!("No ENVIRONMENT_API_KEY set - running without authentication");
    }

    // Load LLM API key for Bearer auth
    let llm_api_key = env::var("GROQ_API_KEY")
        .ok()
        .or_else(|| env::var("OPENAI_API_KEY").ok());

    if llm_api_key.is_some() {
        info!("LLM API key configured");
    } else {
        warn!("No GROQ_API_KEY or OPENAI_API_KEY set - LLM calls may fail");
    }

    // Initialize state
    let state = Arc::new(AppState {
        dataset: Dataset::new(),
        http_client: reqwest::Client::new(),
        api_key,
        llm_api_key,
    });

    info!("Dataset loaded: {} samples", state.dataset.len());

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/task_info", get(task_info_handler))
        .route("/rollout", post(rollout_handler))
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    info!("Starting task app on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
