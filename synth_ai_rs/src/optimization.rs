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
}

impl PolicyOptimizationJobConfig {
    pub fn from_json(config: Value) -> Self {
        Self { config }
    }

    pub fn from_toml_str(input: &str) -> Result<Self> {
        let value: toml::Value =
            toml::from_str(input).map_err(|err| SynthError::UnexpectedResponse(err.to_string()))?;
        let config = serde_json::to_value(value)?;
        Ok(Self { config })
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
        let algorithm = payload
            .get("prompt_learning")
            .and_then(|v| v.get("algorithm"))
            .and_then(|v| v.as_str())
            .unwrap_or("gepa");
        let submit_body = json!({
            "algorithm": algorithm,
            "config_body": payload,
        });
        let resp = client
            .post_json_fallback(
                &[
                    "/policy-optimization/online/jobs",
                    "/prompt-learning/online/jobs",
                ],
                &submit_body,
                AuthStyle::Both,
            )
            .await?;
        let job_id = resp
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SynthError::UnexpectedResponse("missing job_id".to_string()))?;
        Ok(Self::new(client, job_id))
    }

    pub async fn status(&self) -> Result<Value> {
        let path = format!(
            "/policy-optimization/online/jobs/{}",
            self.job_id
        );
        let fallback = format!("/prompt-learning/online/jobs/{}", self.job_id);
        self.client
            .get_json_fallback(
                &[path.as_str(), fallback.as_str()],
                AuthStyle::Both,
            )
            .await
    }

    pub async fn events(&self) -> Result<Vec<Value>> {
        let path = format!(
            "/policy-optimization/online/jobs/{}/events",
            self.job_id
        );
        let fallback = format!("/prompt-learning/online/jobs/{}/events", self.job_id);
        let value = self
            .client
            .get_json_fallback(
                &[path.as_str(), fallback.as_str()],
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
        let primary = format!(
            "{}/policy-optimization/online/jobs/{}/events/stream",
            self.client.api_base(),
            self.job_id
        );
        let fallback = format!(
            "{}/prompt-learning/online/jobs/{}/events/stream",
            self.client.api_base(),
            self.job_id
        );
        let headers = self.client.auth_headers(AuthStyle::Both);
        match stream_sse(self.client.http(), primary, headers.clone()).await {
            Ok(stream) => Ok(stream),
            Err(SynthError::Api { status: 404, .. }) => {
                stream_sse(self.client.http(), fallback, headers).await
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
            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let data = event.get("data").and_then(|v| v.as_object());
            if data.is_none() {
                continue;
            }
            let data = data.unwrap();

            match event_type {
                "learning.policy.gepa.candidate.new_best" => {
                    results.best_prompt = data.get("best_prompt").cloned();
                    if results.best_score.is_none() {
                        results.best_score = extract_reward_value(data, &["best_score"]);
                    }
                }
                "learning.policy.gepa.candidate.evaluated" => {
                    if let Some(rank) = data.get("rank").and_then(|v| v.as_i64()) {
                        let mut prompt_entry = Map::new();
                        prompt_entry.insert("rank".to_string(), json!(rank));
                        prompt_entry.insert(
                            "train_accuracy".to_string(),
                            data.get("train_accuracy").cloned().unwrap_or(Value::Null),
                        );
                        prompt_entry.insert(
                            "val_accuracy".to_string(),
                            data.get("val_accuracy").cloned().unwrap_or(Value::Null),
                        );
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
                }
                "learning.policy.gepa.validation.completed" => {
                    results.validation_results.push(Value::Object(data.clone()));
                    if let (Some(rank), Some(score)) = (
                        data.get("rank").and_then(|v| v.as_i64()),
                        extract_reward_value(data, &[]),
                    ) {
                        validation_by_rank.insert(rank, score);
                    }
                }
                "learning.policy.mipro.job.completed" => {
                    if results.best_score.is_none() {
                        results.best_score = extract_reward_value(
                            data,
                            &["best_score", "best_full_score", "best_minibatch_score"],
                        );
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
