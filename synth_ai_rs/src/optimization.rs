use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::client::{AuthStyle, SynthClient};
use crate::sse::{stream_sse, SseStream};
use crate::types::{Result, SynthError};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Algorithm {
    Gepa,
    Mipro,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyOptimizationJobConfig {
    pub config: Value,
    #[serde(skip_serializing)]
    pub task_app_worker_token: Option<String>,
}

impl PolicyOptimizationJobConfig {
    pub fn from_json(config: Value) -> Self {
        Self {
            config,
            task_app_worker_token: None,
        }
    }

    pub fn from_toml_str(input: &str) -> Result<Self> {
        let value: toml::Value =
            toml::from_str(input).map_err(|err| SynthError::UnexpectedResponse(err.to_string()))?;
        let config = serde_json::to_value(value)?;
        Ok(Self {
            config,
            task_app_worker_token: None,
        })
    }

    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    pub fn to_payload(&self) -> Value {
        let mut config = self.config.clone();
        if let Value::Object(ref mut obj) = config {
            if let Some(policy_opt) = obj.remove("policy_optimization") {
                obj.insert("prompt_learning".to_string(), policy_opt);
            }
            if let Some(Value::Object(pl)) = obj.get_mut("prompt_learning") {
                if let Some(local_url) = pl.remove("localapi_url") {
                    pl.insert("task_app_url".to_string(), local_url);
                }
            }
        }
        config
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptLearningResults {
    pub best_prompt: Option<Value>,
    pub best_score: Option<f64>,
    pub top_prompts: Vec<Value>,
    pub optimized_candidates: Vec<Value>,
    pub attempted_candidates: Vec<Value>,
    pub validation_results: Vec<Value>,
    pub best_candidate: Option<Value>,
    pub event_counts: HashMap<String, i64>,
    pub event_history: Vec<Value>,
    pub gepa: HashMap<String, Value>,
    pub mipro: HashMap<String, Value>,
}

#[derive(Clone)]
pub struct PolicyOptimizationJob {
    client: SynthClient,
    job_id: String,
}

impl PolicyOptimizationJob {
    pub fn new(client: SynthClient, job_id: impl Into<String>) -> Self {
        Self {
            client,
            job_id: job_id.into(),
        }
    }

    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    pub async fn submit(client: SynthClient, config: &PolicyOptimizationJobConfig) -> Result<Self> {
        let payload = config.to_payload();
        let worker_token = config.task_app_worker_token.clone();
        let algorithm = payload
            .get("prompt_learning")
            .and_then(|v| v.get("algorithm"))
            .and_then(|v| v.as_str())
            .unwrap_or("gepa");
        let submit_body = json!({
            "algorithm": algorithm,
            "config_body": payload,
        });
        let resp = if let Some(token) = worker_token {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "X-SynthTunnel-Worker-Token",
                reqwest::header::HeaderValue::from_str(&token).map_err(|_| {
                    SynthError::UnexpectedResponse("invalid SynthTunnel worker token".to_string())
                })?,
            );
            client
                .post_json_fallback_with_headers(
                    &[
                        "/jobs/gepa",
                        "/policy-optimization/online/jobs",
                        "/prompt-learning/online/jobs",
                    ],
                    &submit_body,
                    AuthStyle::Both,
                    Some(headers),
                )
                .await?
        } else {
            client
                .post_json_fallback(
                    &[
                        "/jobs/gepa",
                        "/policy-optimization/online/jobs",
                        "/prompt-learning/online/jobs",
                    ],
                    &submit_body,
                    AuthStyle::Both,
                )
                .await?
        };
        let job_id = resp
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SynthError::UnexpectedResponse("missing job_id".to_string()))?;
        Ok(Self::new(client, job_id))
    }

    pub async fn status(&self) -> Result<Value> {
        let canonical = format!("/jobs/{}", self.job_id);
        let legacy1 = format!("/policy-optimization/online/jobs/{}", self.job_id);
        let legacy2 = format!("/prompt-learning/online/jobs/{}", self.job_id);
        self.client
            .get_json_fallback(
                &[canonical.as_str(), legacy1.as_str(), legacy2.as_str()],
                AuthStyle::Both,
            )
            .await
    }

    pub async fn events(&self) -> Result<Vec<Value>> {
        let canonical = format!("/jobs/{}/events", self.job_id);
        let legacy1 = format!("/policy-optimization/online/jobs/{}/events", self.job_id);
        let legacy2 = format!("/prompt-learning/online/jobs/{}/events", self.job_id);
        let value = self
            .client
            .get_json_fallback(
                &[canonical.as_str(), legacy1.as_str(), legacy2.as_str()],
                AuthStyle::Both,
            )
            .await?;
        parse_events(value)
    }

    pub async fn results(&self) -> Result<PromptLearningResults> {
        let events = self.events().await?;
        Ok(PromptLearningResults::from_events(&events))
    }

    pub async fn stream_events(&self) -> Result<SseStream> {
        let canonical = format!(
            "{}/jobs/{}/events/stream",
            self.client.api_base(),
            self.job_id
        );
        let legacy1 = format!(
            "{}/policy-optimization/online/jobs/{}/events/stream",
            self.client.api_base(),
            self.job_id
        );
        let legacy2 = format!(
            "{}/prompt-learning/online/jobs/{}/events/stream",
            self.client.api_base(),
            self.job_id
        );
        let headers = self.client.auth_headers(AuthStyle::Both);
        match stream_sse(canonical, headers.clone()).await {
            Ok(stream) => Ok(stream),
            Err(SynthError::Api { status: 404, .. }) => {
                match stream_sse(legacy1, headers.clone()).await {
                    Ok(stream) => Ok(stream),
                    Err(SynthError::Api { status: 404, .. }) => {
                        stream_sse(legacy2, headers).await
                    }
                    Err(err) => Err(err),
                }
            }
            Err(err) => Err(err),
        }
    }
}

impl PromptLearningResults {
    pub fn from_events(events: &[Value]) -> Self {
        let mut results = PromptLearningResults::default();
        let mut validation_by_rank: HashMap<i64, f64> = HashMap::new();

        for event in events {
            let event_type = extract_event_type(event);
            let data = match extract_event_data(event) {
                Some(data) => data,
                None => continue,
            };

            if !event_type.is_empty() {
                *results.event_counts.entry(event_type.clone()).or_insert(0) += 1;
                push_event_history(&mut results, &event_type, event, &data);
            }

            match event_type.as_str() {
                "learning.policy.gepa.candidate.new_best" => {
                    results.best_prompt = data.get("best_prompt").cloned();
                    if results.best_score.is_none() {
                        results.best_score = extract_reward_value(data, &["best_score"]);
                    }
                    if results.best_candidate.is_none() {
                        results.best_candidate = data
                            .get("best_candidate")
                            .cloned()
                            .or_else(|| data.get("candidate").cloned())
                            .or_else(|| data.get("program_candidate").cloned());
                    }
                    append_event_bucket(&mut results.gepa, "best_candidates", &event_type, &data);
                }
                "learning.policy.gepa.candidate.evaluated" => {
                    let candidate_view = merge_candidate_payload(data);
                    if let Some(rank) = data.get("rank").and_then(|v| v.as_i64()) {
                        let mut prompt_entry = Map::new();
                        prompt_entry.insert("rank".to_string(), json!(rank));
                        if let Some(val) = candidate_view.get("train_accuracy") {
                            prompt_entry.insert("train_accuracy".to_string(), val.clone());
                        }
                        if let Some(val) = candidate_view.get("val_accuracy") {
                            prompt_entry.insert("val_accuracy".to_string(), val.clone());
                        }
                        if let Some(candidate_id) = candidate_view
                            .get("candidate_id")
                            .or_else(|| candidate_view.get("version_id"))
                        {
                            prompt_entry.insert("candidate_id".to_string(), candidate_id.clone());
                        }
                        for key in [
                            "parent_id",
                            "generation",
                            "accepted",
                            "is_pareto",
                            "mutation_type",
                            "mutation_params",
                            "prompt_summary",
                            "prompt_text",
                            "objectives",
                            "instance_scores",
                            "instance_objectives",
                            "seed_scores",
                            "seed_info",
                            "rollout_sample",
                            "token_usage",
                            "cost_usd",
                            "evaluation_duration_ms",
                            "skip_reason",
                            "stages",
                            "seeds_evaluated",
                            "full_score",
                        ] {
                            if let Some(val) = candidate_view.get(key) {
                                prompt_entry.insert(key.to_string(), val.clone());
                            }
                        }
                        if let Some(mutation_type) = candidate_view
                            .get("mutation_type")
                            .or_else(|| candidate_view.get("operator"))
                        {
                            prompt_entry.insert("mutation_type".to_string(), mutation_type.clone());
                        }
                        if let Some(scores) = candidate_view.get("minibatch_scores") {
                            prompt_entry.insert("minibatch_scores".to_string(), scores.clone());
                        } else if let Some(score) = candidate_view.get("minibatch_score") {
                            prompt_entry.insert(
                                "minibatch_scores".to_string(),
                                Value::Array(vec![score.clone()]),
                            );
                        }
                        if let Some(pattern) = data.get("pattern") {
                            prompt_entry.insert("pattern".to_string(), pattern.clone());
                            if let Some(text) = extract_full_text_from_pattern(pattern) {
                                prompt_entry.insert("full_text".to_string(), json!(text));
                            }
                        } else if let Some(template) = data.get("template") {
                            if let Some(pattern) = convert_template_to_pattern(template) {
                                prompt_entry.insert("pattern".to_string(), pattern.clone());
                                if let Some(text) = extract_full_text_from_pattern(&pattern) {
                                    prompt_entry.insert("full_text".to_string(), json!(text));
                                }
                            }
                        }
                        results.top_prompts.push(Value::Object(prompt_entry));
                    }
                    append_event_bucket(&mut results.gepa, "candidates", &event_type, &candidate_view);
                }
                "learning.policy.gepa.job.completed" => {
                    if let Some(cands) = data.get("optimized_candidates").and_then(|v| v.as_array())
                    {
                        results.optimized_candidates = cands.clone();
                    }
                    if let Some(cands) = data.get("attempted_candidates").and_then(|v| v.as_array())
                    {
                        results.attempted_candidates = cands.clone();
                    }
                    if results.best_prompt.is_none() {
                        results.best_prompt = data.get("best_prompt").cloned();
                    }
                    if results.best_score.is_none() {
                        results.best_score = extract_reward_value(data, &["best_score"]);
                    }

                    if let Some(validation) = data.get("validation").and_then(|v| v.as_array()) {
                        for val in validation {
                            if let Some(val_obj) = val.as_object() {
                                if let (Some(rank), Some(score)) = (
                                    val_obj.get("rank").and_then(|v| v.as_i64()),
                                    extract_reward_value(val_obj, &[]),
                                ) {
                                    validation_by_rank.insert(rank, score);
                                }
                            }
                        }
                    }
                    if let Some(Value::Object(baseline)) = data.get("baseline") {
                        results.gepa.insert("baseline".to_string(), Value::Object(baseline.clone()));
                    }
                    append_event_bucket(&mut results.gepa, "job_completed", &event_type, &data);
                }
                "learning.policy.gepa.validation.completed" => {
                    results.validation_results.push(Value::Object(data.clone()));
                    if let (Some(rank), Some(score)) = (
                        data.get("rank").and_then(|v| v.as_i64()),
                        extract_reward_value(data, &[]),
                    ) {
                        validation_by_rank.insert(rank, score);
                    }
                    append_event_bucket(&mut results.gepa, "validation", &event_type, &data);
                }
                "learning.policy.mipro.job.completed" => {
                    if results.best_score.is_none() {
                        results.best_score = extract_reward_value(
                            data,
                            &["best_score", "best_full_score", "best_minibatch_score"],
                        );
                    }
                    if let Some(cands) = data.get("attempted_candidates").and_then(|v| v.as_array())
                    {
                        results.attempted_candidates = cands.clone();
                    }
                    if let Some(cands) = data.get("optimized_candidates").and_then(|v| v.as_array())
                    {
                        results.optimized_candidates = cands.clone();
                    }
                    let mut entry = Map::new();
                    entry.insert(
                        "_event_type".to_string(),
                        Value::String(event_type.clone()),
                    );
                    for (k, v) in data.iter() {
                        entry.insert(k.clone(), v.clone());
                    }
                    results.mipro.insert("job".to_string(), Value::Object(entry));
                }
                _ if event_type.starts_with("learning.policy.gepa.") => {
                    if event_type.contains("baseline") {
                        results
                            .gepa
                            .insert("baseline".to_string(), Value::Object(data.clone()));
                    } else if event_type.contains("frontier_updated")
                        || event_type.contains("frontier.updated")
                    {
                        append_event_bucket(&mut results.gepa, "frontier_updates", &event_type, &data);
                    } else if event_type.contains("generation.complete")
                        || event_type.contains("generation.completed")
                    {
                        append_event_bucket(&mut results.gepa, "generations", &event_type, &data);
                    } else if event_type.contains("progress") {
                        append_event_bucket(&mut results.gepa, "progress_updates", &event_type, &data);
                    }
                }
                _ if event_type.starts_with("learning.policy.mipro.") => {
                    if event_type.contains("iteration.started")
                        || event_type.contains("iteration.completed")
                    {
                        append_event_bucket(&mut results.mipro, "iterations", &event_type, &data);
                    } else if event_type.contains("trial.started")
                        || event_type.contains("trial.completed")
                    {
                        append_event_bucket(&mut results.mipro, "trials", &event_type, &data);
                    } else if event_type.contains("candidate.new_best") {
                        append_event_bucket(&mut results.mipro, "incumbents", &event_type, &data);
                        if let Some(score) = extract_reward_value(
                            data,
                            &["best_score", "score", "full_score", "minibatch_score"],
                        ) {
                            if results.best_score.is_none()
                                || results.best_score.map(|s| score > s).unwrap_or(false)
                            {
                                results.best_score = Some(score);
                            }
                        }
                        if results.best_candidate.is_none() {
                            results.best_candidate = data
                                .get("best_candidate")
                                .cloned()
                                .or_else(|| data.get("candidate").cloned())
                                .or_else(|| data.get("program_candidate").cloned());
                        }
                    } else if event_type.contains("candidate.evaluated") {
                        append_event_bucket(&mut results.mipro, "candidates", &event_type, &data);
                    } else if event_type.contains("budget") {
                        append_event_bucket(&mut results.mipro, "budget_updates", &event_type, &data);
                    }
                }
                _ => {}
            }
        }

        if results.top_prompts.is_empty() && !results.optimized_candidates.is_empty() {
            for (idx, cand) in results.optimized_candidates.iter().enumerate() {
                let cand_obj = match cand.as_object() {
                    Some(obj) => obj,
                    None => continue,
                };
                let rank = cand_obj
                    .get("rank")
                    .and_then(|v| v.as_i64())
                    .unwrap_or((idx + 1) as i64);
                let mut prompt_entry = Map::new();
                prompt_entry.insert("rank".to_string(), json!(rank));

                let train_accuracy = cand_obj
                    .get("score")
                    .and_then(|v| v.as_object())
                    .and_then(|v| extract_reward_value(v, &[]))
                    .or_else(|| extract_reward_value(cand_obj, &[]));
                if let Some(score) = train_accuracy {
                    prompt_entry.insert("train_accuracy".to_string(), json!(score));
                }
                if let Some(val) = validation_by_rank.get(&rank) {
                    prompt_entry.insert("val_accuracy".to_string(), json!(*val));
                }
                for key in [
                    "candidate_id",
                    "version_id",
                    "parent_id",
                    "generation",
                    "accepted",
                    "is_pareto",
                    "mutation_type",
                    "mutation_params",
                    "prompt_summary",
                    "prompt_text",
                    "objectives",
                    "instance_scores",
                    "instance_objectives",
                    "seed_scores",
                    "seed_info",
                    "rollout_sample",
                    "token_usage",
                    "cost_usd",
                    "evaluation_duration_ms",
                    "skip_reason",
                    "stages",
                    "seeds_evaluated",
                    "full_score",
                ] {
                    if let Some(val) = cand_obj.get(key) {
                        prompt_entry.insert(key.to_string(), val.clone());
                    }
                }

                if let Some(pattern) = cand_obj.get("pattern") {
                    prompt_entry.insert("pattern".to_string(), pattern.clone());
                    if let Some(text) = extract_full_text_from_pattern(pattern) {
                        prompt_entry.insert("full_text".to_string(), json!(text));
                    }
                } else if let Some(template) = cand_obj.get("template") {
                    if let Some(pattern) = convert_template_to_pattern(template) {
                        if let Some(text) = extract_full_text_from_pattern(&pattern) {
                            prompt_entry.insert("full_text".to_string(), json!(text));
                        }
                        prompt_entry.insert("pattern".to_string(), pattern);
                    }
                }

                results.top_prompts.push(Value::Object(prompt_entry));
            }
        }

        results
    }
}

fn parse_events(value: Value) -> Result<Vec<Value>> {
    if let Value::Array(items) = value {
        return Ok(items);
    }
    if let Value::Object(obj) = value {
        if let Some(Value::Array(items)) = obj.get("events") {
            return Ok(items.clone());
        }
    }
    Err(SynthError::UnexpectedResponse(
        "events response did not contain an events list".to_string(),
    ))
}

fn coerce_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(num) => num.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn extract_outcome_reward(payload: &Map<String, Value>) -> Option<f64> {
    if let Some(Value::Object(obj)) = payload.get("outcome_objectives") {
        if let Some(val) = obj.get("reward").and_then(coerce_f64) {
            return Some(val);
        }
    }
    payload.get("outcome_reward").and_then(coerce_f64)
}

fn extract_reward_value(payload: &Map<String, Value>, fallback_keys: &[&str]) -> Option<f64> {
    if let Some(val) = extract_outcome_reward(payload) {
        return Some(val);
    }
    for key in fallback_keys {
        if let Some(val) = payload.get(*key).and_then(coerce_f64) {
            return Some(val);
        }
    }
    None
}

fn extract_event_type(event: &Value) -> String {
    event
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| event.get("event_type").and_then(|v| v.as_str()))
        .or_else(|| {
            event
                .get("data")
                .and_then(|v| v.as_object())
                .and_then(|obj| obj.get("type").and_then(|v| v.as_str()))
        })
        .or_else(|| {
            event
                .get("data")
                .and_then(|v| v.as_object())
                .and_then(|obj| obj.get("event_type").and_then(|v| v.as_str()))
        })
        .unwrap_or("")
        .to_string()
}

fn extract_event_data(event: &Value) -> Option<Map<String, Value>> {
    let mut data = event
        .get("data")
        .cloned()
        .or_else(|| event.get("payload").cloned())
        .or_else(|| event.get("data_json").cloned())?;
    let mut data = match data {
        Value::Object(obj) => obj,
        _ => return None,
    };

    loop {
        let looks_wrapped = data.contains_key("data")
            && data.keys().any(|k| {
                matches!(
                    k.as_str(),
                    "type" | "event_type" | "message" | "seq" | "timestamp_ms" | "ts"
                )
            });
        if !looks_wrapped {
            break;
        }
        let inner = data.get("data");
        if let Some(Value::Object(inner_obj)) = inner {
            data = inner_obj.clone();
            continue;
        }
        break;
    }

    Some(data)
}

fn append_event_bucket(
    bucket: &mut HashMap<String, Value>,
    key: &str,
    event_type: &str,
    data: &Map<String, Value>,
) {
    if data.is_empty() {
        return;
    }
    let mut entry = Map::new();
    entry.insert(
        "_event_type".to_string(),
        Value::String(event_type.to_string()),
    );
    for (k, v) in data {
        entry.insert(k.clone(), v.clone());
    }
    let slot = bucket
        .entry(key.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(items) = slot {
        items.push(Value::Object(entry));
    }
}

fn push_event_history(
    results: &mut PromptLearningResults,
    event_type: &str,
    event: &Value,
    data: &Map<String, Value>,
) {
    let mut entry = Map::new();
    if !event_type.is_empty() {
        entry.insert("type".to_string(), Value::String(event_type.to_string()));
    }
    if let Some(seq) = event.get("seq") {
        entry.insert("seq".to_string(), seq.clone());
    }
    if let Some(ts) = event.get("timestamp_ms") {
        entry.insert("timestamp_ms".to_string(), ts.clone());
    }
    if let Some(ts) = event.get("ts") {
        entry.insert("ts".to_string(), ts.clone());
    }
    if let Some(msg) = event.get("message") {
        entry.insert("message".to_string(), msg.clone());
    }
    entry.insert("data".to_string(), Value::Object(data.clone()));
    results.event_history.push(Value::Object(entry));
}

fn merge_candidate_payload(data: &Map<String, Value>) -> Map<String, Value> {
    let mut merged = data.clone();
    if let Some(Value::Object(program_candidate)) = data.get("program_candidate") {
        for (k, v) in program_candidate {
            merged.insert(k.clone(), v.clone());
        }
    }
    merged
}

fn convert_template_to_pattern(template: &Value) -> Option<Value> {
    let sections = template
        .get("sections")
        .and_then(|v| v.as_array())
        .filter(|v| !v.is_empty())
        .or_else(|| template.get("prompt_sections").and_then(|v| v.as_array()))?;
    let mut messages = Vec::new();
    for sec in sections {
        let sec_obj = sec.as_object()?;
        let content = sec_obj.get("content")?;
        if content.is_null() {
            continue;
        }
        let role = sec_obj
            .get("role")
            .and_then(|v| v.as_str())
            .or_else(|| sec_obj.get("name").and_then(|v| v.as_str()))
            .unwrap_or("system");
        let name = sec_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
        messages.push(json!({
            "role": role,
            "name": name,
            "pattern": content,
        }));
    }
    if messages.is_empty() {
        return None;
    }
    Some(json!({ "messages": messages }))
}

fn extract_full_text_from_pattern(pattern: &Value) -> Option<String> {
    let messages = pattern.get("messages")?.as_array()?;
    let mut parts = Vec::new();
    for msg in messages {
        let msg_obj = msg.as_object()?;
        let role = msg_obj.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let name = msg_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let content = msg_obj
            .get("pattern")
            .or_else(|| msg_obj.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        parts.push(format!("[{role} | {name}]\n{content}"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}
