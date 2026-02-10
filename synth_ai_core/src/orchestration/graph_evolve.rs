//! Graph Evolve (GraphGen) helpers.
//!
//! These helpers normalize datasets/configs and build payloads so
//! both Rust and Python SDKs share the same core logic.

use serde_json::{json, Map, Value};

use crate::data::enums::GraphType;
use crate::errors::CoreError;

const DEFAULT_POPULATION_SIZE: i64 = 4;
const DEFAULT_NUM_PARENTS: i64 = 2;
const DEFAULT_ROLLOUT_MAX_CONCURRENT: i64 = 25;
const DEFAULT_ROLLOUT_TIMEOUT_SECONDS: f64 = 60.0;

fn parse_graph_type(value: Option<&str>) -> Result<GraphType, CoreError> {
    let Some(raw) = value else {
        return Ok(GraphType::Policy);
    };
    match raw.trim().to_lowercase().as_str() {
        "policy" => Ok(GraphType::Policy),
        "verifier" => Ok(GraphType::Verifier),
        "rlm" => Ok(GraphType::Rlm),
        other => Err(CoreError::Validation(format!(
            "invalid graph_type '{}'; expected 'policy', 'verifier', or 'rlm'",
            other
        ))),
    }
}

fn as_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(|v| v.as_i64())
}

fn ensure_task_list(dataset: &Map<String, Value>) -> Result<(), CoreError> {
    match dataset.get("tasks") {
        Some(Value::Array(tasks)) if !tasks.is_empty() => Ok(()),
        Some(Value::Array(_)) => Err(CoreError::Validation(
            "dataset must contain at least one task".to_string(),
        )),
        _ => Err(CoreError::Validation(
            "dataset.tasks must be a non-empty list".to_string(),
        )),
    }
}

/// Parse and validate a Graph Evolve dataset JSON object.
pub fn parse_graph_evolve_dataset(dataset: &Value) -> Result<Value, CoreError> {
    let dataset_map = dataset
        .as_object()
        .ok_or_else(|| CoreError::Validation("dataset must be an object".to_string()))?;
    ensure_task_list(dataset_map)?;
    Ok(Value::Object(dataset_map.clone()))
}

/// Load and validate a Graph Evolve dataset from a JSON file.
pub fn load_graph_evolve_dataset(path: &str) -> Result<Value, CoreError> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        CoreError::InvalidInput(format!("failed to read dataset file '{}': {}", path, e))
    })?;
    let value: Value = serde_json::from_str(&contents).map_err(|e| {
        CoreError::Validation(format!("failed to parse dataset JSON '{}': {}", path, e))
    })?;
    parse_graph_evolve_dataset(&value)
}

/// Ensure policy model list is non-empty.
pub fn normalize_graph_evolve_policy_models(models: Vec<String>) -> Result<Vec<String>, CoreError> {
    let filtered: Vec<String> = models
        .into_iter()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .collect();
    if filtered.is_empty() {
        return Err(CoreError::Validation(
            "policy_models must contain at least one model".to_string(),
        ));
    }
    Ok(filtered)
}

/// Build a Graph Evolve config dict with defaults.
#[allow(clippy::too_many_arguments)]
pub fn build_graph_evolve_config(
    policy_models: Vec<String>,
    rollout_budget: i64,
    proposer_effort: &str,
    verifier_model: Option<String>,
    verifier_provider: Option<String>,
    population_size: i64,
    num_generations: Option<i64>,
    problem_spec: Option<String>,
    target_llm_calls: Option<i64>,
    graph_type: Option<String>,
    initial_graph_id: Option<String>,
) -> Result<Value, CoreError> {
    let policy_models = normalize_graph_evolve_policy_models(policy_models)?;

    if rollout_budget < 10 || rollout_budget > 10000 {
        return Err(CoreError::Validation(format!(
            "rollout_budget must be between 10 and 10000, got {}",
            rollout_budget
        )));
    }

    let effort = proposer_effort.trim().to_lowercase();
    if effort != "low" && effort != "medium" && effort != "high" {
        return Err(CoreError::Validation(
            "proposer_effort must be one of: low, medium, high".to_string(),
        ));
    }

    if population_size < 2 || population_size > 20 {
        return Err(CoreError::Validation(format!(
            "population_size must be between 2 and 20, got {}",
            population_size
        )));
    }

    if let Some(value) = num_generations {
        if value < 1 || value > 50 {
            return Err(CoreError::Validation(format!(
                "num_generations must be between 1 and 50, got {}",
                value
            )));
        }
    }

    if let Some(value) = target_llm_calls {
        if value < 1 || value > 10 {
            return Err(CoreError::Validation(format!(
                "target_llm_calls must be between 1 and 10, got {}",
                value
            )));
        }
    }

    let initial_graph_id = initial_graph_id.ok_or_else(|| {
        CoreError::Validation(
            "initial_graph_id is required for Graph Evolve (de-novo graph generation is disabled)"
                .to_string(),
        )
    })?;

    let graph_type = parse_graph_type(graph_type.as_deref())?;

    let mut map = Map::new();
    map.insert("graph_type".to_string(), json!(graph_type));
    map.insert("policy_models".to_string(), json!(policy_models));
    map.insert("rollout_budget".to_string(), json!(rollout_budget));
    map.insert(
        "rollout_max_concurrent".to_string(),
        json!(DEFAULT_ROLLOUT_MAX_CONCURRENT),
    );
    map.insert(
        "rollout_timeout_seconds".to_string(),
        json!(DEFAULT_ROLLOUT_TIMEOUT_SECONDS),
    );
    map.insert("proposer_effort".to_string(), json!(effort));
    map.insert("population_size".to_string(), json!(population_size));
    map.insert("num_parents".to_string(), json!(DEFAULT_NUM_PARENTS));
    map.insert("initial_graph_id".to_string(), json!(initial_graph_id));

    if let Some(value) = verifier_model {
        map.insert("verifier_model".to_string(), json!(value));
    }
    if let Some(value) = verifier_provider {
        map.insert("verifier_provider".to_string(), json!(value));
    }
    if let Some(value) = num_generations {
        map.insert("num_generations".to_string(), json!(value));
    }
    if let Some(value) = problem_spec {
        map.insert("problem_spec".to_string(), json!(value));
    }
    if let Some(value) = target_llm_calls {
        map.insert("target_llm_calls".to_string(), json!(value));
    }

    Ok(Value::Object(map))
}

/// Build a Graph Evolve payload.
pub fn build_graph_evolve_payload(
    dataset: &Value,
    config: &Value,
    metadata: Option<&Value>,
    auto_start: bool,
) -> Result<Value, CoreError> {
    let mut dataset_map = dataset
        .as_object()
        .ok_or_else(|| CoreError::Validation("dataset must be an object".to_string()))?
        .clone();
    ensure_task_list(&dataset_map)?;

    let config_map = config
        .as_object()
        .ok_or_else(|| CoreError::Validation("config must be an object".to_string()))?;

    if !dataset_map.contains_key("initial_prompt") {
        let fallback = config_map
            .get("problem_spec")
            .and_then(|v| v.as_str())
            .unwrap_or("Optimizing prompt graph...");
        dataset_map.insert(
            "initial_prompt".to_string(),
            Value::String(fallback.to_string()),
        );
    }

    let mut metadata_map = match metadata {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    };

    if let Some(value) = as_i64(config_map.get("num_generations")) {
        metadata_map.insert("num_generations".to_string(), json!(value));
    }
    if let Some(value) = as_i64(config_map.get("population_size")) {
        if value != DEFAULT_POPULATION_SIZE {
            metadata_map.insert("population_size".to_string(), json!(value));
        }
    }
    if let Some(value) = as_i64(config_map.get("num_parents")) {
        if value != DEFAULT_NUM_PARENTS {
            metadata_map.insert("num_parents".to_string(), json!(value));
        }
    }
    if let Some(Value::Array(seeds)) = config_map.get("evaluation_seeds") {
        metadata_map.insert("evaluation_seeds".to_string(), Value::Array(seeds.clone()));
    }

    let eval_sample_size = metadata_map.remove("eval_sample_size");
    let feedback_sample_size = metadata_map.remove("feedback_sample_size");

    let policy_models = config_map
        .get("policy_models")
        .ok_or_else(|| CoreError::Validation("policy_models missing from config".to_string()))?
        .clone();
    let rollout_budget = config_map
        .get("rollout_budget")
        .ok_or_else(|| CoreError::Validation("rollout_budget missing from config".to_string()))?
        .clone();
    let proposer_effort = config_map
        .get("proposer_effort")
        .ok_or_else(|| CoreError::Validation("proposer_effort missing from config".to_string()))?
        .clone();

    let mut payload = Map::new();
    payload.insert("dataset".to_string(), Value::Object(dataset_map));
    payload.insert("initial_prompt".to_string(), Value::Null);
    payload.insert("policy_models".to_string(), policy_models);
    payload.insert("rollout_budget".to_string(), rollout_budget);
    payload.insert("proposer_effort".to_string(), proposer_effort);

    if let Some(value) = config_map.get("policy_provider") {
        if !value.is_null() {
            payload.insert("policy_provider".to_string(), value.clone());
        }
    }
    if let Some(value) = config_map.get("verifier_model") {
        if !value.is_null() {
            payload.insert("judge_model".to_string(), value.clone());
        }
    }
    if let Some(value) = config_map.get("verifier_provider") {
        if !value.is_null() {
            payload.insert("judge_provider".to_string(), value.clone());
        }
    }
    if let Some(value) = config_map.get("problem_spec") {
        if !value.is_null() {
            payload.insert("problem_spec".to_string(), value.clone());
        }
    }
    if let Some(value) = config_map.get("target_llm_calls") {
        if !value.is_null() {
            payload.insert("target_llm_calls".to_string(), value.clone());
        }
    }
    if let Some(value) = config_map.get("initial_graph_id") {
        if !value.is_null() {
            payload.insert("initial_graph_id".to_string(), value.clone());
        } else {
            return Err(CoreError::Validation(
                "initial_graph_id missing from config".to_string(),
            ));
        }
    } else {
        return Err(CoreError::Validation(
            "initial_graph_id missing from config".to_string(),
        ));
    }

    if let Some(value) = eval_sample_size {
        payload.insert("eval_sample_size".to_string(), value);
    }
    if let Some(value) = feedback_sample_size {
        payload.insert("feedback_sample_size".to_string(), value);
    }

    payload.insert("metadata".to_string(), Value::Object(metadata_map));
    payload.insert("auto_start".to_string(), Value::Bool(auto_start));

    Ok(Value::Object(payload))
}

/// Resolve prompt/graph snapshot IDs for graph-evolve requests.
pub fn resolve_graph_evolve_snapshot_id(
    prompt_snapshot_id: Option<&str>,
    graph_snapshot_id: Option<&str>,
) -> Result<Option<String>, CoreError> {
    if prompt_snapshot_id.is_some() && graph_snapshot_id.is_some() {
        return Err(CoreError::Validation(
            "Provide only one of prompt_snapshot_id or graph_snapshot_id.".to_string(),
        ));
    }
    Ok(graph_snapshot_id
        .map(|s| s.to_string())
        .or_else(|| prompt_snapshot_id.map(|s| s.to_string())))
}

/// Build payload for graph record download.
pub fn build_graph_evolve_graph_record_payload(
    job_id: &str,
    prompt_snapshot_id: Option<&str>,
    graph_snapshot_id: Option<&str>,
) -> Result<Value, CoreError> {
    let snapshot_id = resolve_graph_evolve_snapshot_id(prompt_snapshot_id, graph_snapshot_id)?;
    let mut map = Map::new();
    map.insert("job_id".to_string(), Value::String(job_id.to_string()));
    if let Some(snapshot_id) = snapshot_id {
        map.insert("prompt_snapshot_id".to_string(), Value::String(snapshot_id));
    }
    Ok(Value::Object(map))
}

/// Build payload for graph inference.
pub fn build_graph_evolve_inference_payload(
    job_id: &str,
    input: &Value,
    model: Option<&str>,
    prompt_snapshot_id: Option<&str>,
    graph_snapshot_id: Option<&str>,
) -> Result<Value, CoreError> {
    let snapshot_id = resolve_graph_evolve_snapshot_id(prompt_snapshot_id, graph_snapshot_id)?;
    let mut map = Map::new();
    map.insert("job_id".to_string(), Value::String(job_id.to_string()));
    map.insert("input".to_string(), input.clone());
    if let Some(model) = model {
        map.insert("model".to_string(), Value::String(model.to_string()));
    }
    if let Some(snapshot_id) = snapshot_id {
        map.insert("prompt_snapshot_id".to_string(), Value::String(snapshot_id));
    }
    Ok(Value::Object(map))
}

/// Build a placeholder dataset for resumed jobs.
pub fn build_graph_evolve_placeholder_dataset() -> Value {
    json!({
        "metadata": {"name": "(resumed job)"},
        "tasks": [{"id": "placeholder", "input": {}}]
    })
}

// =============================================================================
// Graph Evolve job orchestration
// =============================================================================

use crate::api::SynthClient;

/// High-level Graph Evolve job orchestration.
pub struct GraphEvolveJob {
    client: SynthClient,
    job_id: Option<String>,
    legacy_graphgen_job_id: Option<String>,
    payload: Option<Value>,
}

impl GraphEvolveJob {
    /// Create a job from a payload.
    pub fn from_payload(
        payload: Value,
        api_key: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Self, CoreError> {
        let api_key = match api_key {
            Some(k) => k.to_string(),
            None => crate::auth::get_api_key(None)
                .ok_or_else(|| CoreError::Authentication("SYNTH_API_KEY not found".to_string()))?,
        };

        let client = SynthClient::new(&api_key, base_url)?;

        Ok(Self {
            client,
            job_id: None,
            legacy_graphgen_job_id: None,
            payload: Some(payload),
        })
    }

    /// Reconnect to an existing job by ID.
    pub fn from_job_id(
        job_id: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Self, CoreError> {
        let api_key = match api_key {
            Some(k) => k.to_string(),
            None => crate::auth::get_api_key(None)
                .ok_or_else(|| CoreError::Authentication("SYNTH_API_KEY not found".to_string()))?,
        };

        let client = SynthClient::new(&api_key, base_url)?;

        let mut job = Self {
            client,
            job_id: Some(job_id.to_string()),
            legacy_graphgen_job_id: None,
            payload: None,
        };

        if job_id.starts_with("graphgen_") {
            job.legacy_graphgen_job_id = Some(job_id.to_string());
        }

        Ok(job)
    }

    /// Get the job ID (primary or legacy).
    pub fn job_id(&self) -> Option<&str> {
        if let Some(id) = self.job_id.as_deref() {
            return Some(id);
        }
        self.legacy_graphgen_job_id.as_deref()
    }

    /// Get the legacy GraphGen job ID, if known.
    pub fn legacy_job_id(&self) -> Option<&str> {
        self.legacy_graphgen_job_id.as_deref()
    }

    fn require_job_id(&self) -> Result<&str, CoreError> {
        self.job_id()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))
    }

    /// Submit the job and return the backend response.
    pub async fn submit(&mut self) -> Result<Value, CoreError> {
        if self.job_id.is_some() || self.legacy_graphgen_job_id.is_some() {
            return Err(CoreError::Validation("job already submitted".to_string()));
        }
        let payload = self
            .payload
            .as_ref()
            .ok_or_else(|| CoreError::Validation("payload missing".to_string()))?;
        let response = self
            .client
            .graph_evolve()
            .submit_job(payload.clone())
            .await?;

        if let Some(id) = response.get("graph_evolve_job_id").and_then(|v| v.as_str()) {
            self.job_id = Some(id.to_string());
        }
        if let Some(id) = response.get("graphgen_job_id").and_then(|v| v.as_str()) {
            self.legacy_graphgen_job_id = Some(id.to_string());
            if self.job_id.is_none() {
                self.job_id = Some(id.to_string());
            }
        }

        Ok(response)
    }

    /// Get current job status.
    pub async fn get_status(&self) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client.graph_evolve().get_status(job_id).await
    }

    /// Start a queued job.
    pub async fn start(&self) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client.graph_evolve().start_job(job_id).await
    }

    /// Fetch events for the job.
    pub async fn get_events(&self, since_seq: i64, limit: i64) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client
            .graph_evolve()
            .get_events(job_id, since_seq, limit)
            .await
    }

    /// Fetch metrics for the job.
    pub async fn get_metrics(&self, query_string: &str) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client
            .graph_evolve()
            .get_metrics(job_id, query_string)
            .await
    }

    /// Download prompt (JSON response).
    pub async fn download_prompt(&self) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client.graph_evolve().download_prompt(job_id).await
    }

    /// Download redacted graph export.
    pub async fn download_graph_txt(&self) -> Result<String, CoreError> {
        let job_id = self.require_job_id()?;
        self.client.graph_evolve().download_graph_txt(job_id).await
    }

    /// Run inference using the optimized graph.
    pub async fn run_inference(&self, payload: Value) -> Result<Value, CoreError> {
        self.client.graph_evolve().run_inference(payload).await
    }

    /// Fetch a graph record snapshot.
    pub async fn get_graph_record(&self, payload: Value) -> Result<Value, CoreError> {
        self.client.graph_evolve().get_graph_record(payload).await
    }

    /// Cancel the job.
    pub async fn cancel(&self, payload: Value) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client.graph_evolve().cancel_job(job_id, payload).await
    }

    /// Query workflow state.
    pub async fn query_workflow_state(&self) -> Result<Value, CoreError> {
        let job_id = self.require_job_id()?;
        self.client
            .graph_evolve()
            .query_workflow_state(job_id)
            .await
    }
}
