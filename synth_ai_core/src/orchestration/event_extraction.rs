use serde_json::{Map, Value};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::progress::{SeedInfo, StageInfo, TokenUsage};
use crate::CoreError;

fn value_to_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .or_else(|| value.as_u64().map(|v| v as f64))
        .or_else(|| value.as_str().and_then(|s| s.parse::<f64>().ok()))
}

fn value_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| value.as_f64().map(|v| v as i64))
        .or_else(|| value.as_str().and_then(|s| s.parse::<i64>().ok()))
}

fn extract_instruction_text(value: &Value) -> String {
    if value.is_null() {
        return String::new();
    }

    if let Some(text) = value.as_str() {
        let trimmed = text.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                if let Some(obj) = parsed.as_object() {
                    for key in ["instruction", "text", "content", "prompt"] {
                        if let Some(Value::String(val)) = obj.get(key) {
                            return val.clone();
                        }
                    }
                    for (_, value) in obj {
                        if let Some(val) = value.as_str() {
                            if val.len() > 10 {
                                return val.to_string();
                            }
                        }
                    }
                }
            }
        }
        return text.to_string();
    }

    if let Some(obj) = value.as_object() {
        for key in ["instruction", "text", "content", "prompt"] {
            if let Some(Value::String(val)) = obj.get(key) {
                return val.clone();
            }
        }
    }

    value.to_string()
}

pub fn seed_reward_entry(seed: i64, score: Option<&Value>) -> Value {
    let mut map = Map::new();
    map.insert("seed".to_string(), Value::Number(seed.into()));

    match score {
        None => {
            map.insert("reward".to_string(), Value::Null);
            map.insert("status".to_string(), Value::String("failed".to_string()));
        }
        Some(value) if value.is_null() => {
            map.insert("reward".to_string(), Value::Null);
            map.insert("status".to_string(), Value::String("failed".to_string()));
        }
        Some(value) => {
            if let Some(num) = value_to_f64(value) {
                if num.is_nan() {
                    map.insert("reward".to_string(), Value::Null);
                    map.insert("status".to_string(), Value::String("invalid".to_string()));
                } else if let Some(json_num) = serde_json::Number::from_f64(num) {
                    map.insert("reward".to_string(), Value::Number(json_num));
                } else {
                    map.insert("reward".to_string(), Value::Null);
                    map.insert("status".to_string(), Value::String("invalid".to_string()));
                }
            } else {
                map.insert("reward".to_string(), Value::Null);
                map.insert("status".to_string(), Value::String("invalid".to_string()));
            }
        }
    }

    Value::Object(map)
}

/// Backwards-compatible alias for `seed_reward_entry`.
pub fn seed_score_entry(seed: i64, score: Option<&Value>) -> Value {
    seed_reward_entry(seed, score)
}

fn collect_text_replacements(candidate: &Value, obj: &Map<String, Value>) -> Vec<Value> {
    if let Some(arr) = obj.get("text_replacements").and_then(|v| v.as_array()) {
        return arr.clone();
    }
    if let Some(arr) = obj
        .get("data")
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("text_replacements"))
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    if let Some(arr) = candidate
        .get("transformation")
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("text_replacements"))
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    if let Some(arr) = candidate
        .get("text_replacements")
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    Vec::new()
}

fn collect_messages(candidate: &Value, obj: &Map<String, Value>) -> Vec<Value> {
    if let Some(arr) = candidate
        .get("pattern")
        .or_else(|| obj.get("pattern"))
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("messages"))
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    if let Some(arr) = candidate
        .get("messages")
        .or_else(|| obj.get("messages"))
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    if let Some(arr) = candidate
        .get("transformation")
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("messages"))
        .and_then(|v| v.as_array())
    {
        return arr.clone();
    }
    Vec::new()
}

pub fn extract_stages_from_candidate(
    candidate: &Value,
    require_stages: bool,
    candidate_id: Option<&str>,
) -> Result<Option<HashMap<String, StageInfo>>, CoreError> {
    let obj = candidate
        .get("object")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let cid = candidate_id
        .map(|s| s.to_string())
        .or_else(|| {
            candidate
                .get("version_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let mut stages: HashMap<String, StageInfo> = HashMap::new();

    if let Some(genome) = candidate
        .get("genome")
        .or_else(|| obj.get("genome"))
        .and_then(|v| v.as_object())
    {
        let mut keys: Vec<&String> = genome.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(gene) = genome.get(key).and_then(|v| v.as_object()) {
                let instruction_lines = gene.get("instruction_lines").and_then(|v| v.as_array());
                if let Some(lines) = instruction_lines {
                    if !lines.is_empty() {
                        let instruction = lines
                            .iter()
                            .map(|line| match line.as_str() {
                                Some(text) => text.to_string(),
                                None => line.to_string(),
                            })
                            .collect::<Vec<String>>()
                            .join("\n");
                        let rules = gene
                            .get("rules")
                            .and_then(|v| v.as_object())
                            .map(|map| map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();
                        let temperature = gene.get("temperature").and_then(|v| v.as_f64());
                        stages.insert(
                            key.clone(),
                            StageInfo {
                                instruction,
                                rules,
                                temperature,
                                prompts: None,
                            },
                        );
                    }
                }
            }
        }
    }

    if stages.is_empty() {
        let replacements = collect_text_replacements(candidate, &obj);
        if !replacements.is_empty() {
            let mut role_instructions: HashMap<String, Vec<String>> = HashMap::new();
            for item in replacements {
                if let Some(repl) = item.as_object() {
                    let new_text = repl.get("new_text");
                    let role = repl
                        .get("apply_to_role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("system")
                        .to_lowercase();
                    if let Some(text) = new_text {
                        let cleaned = extract_instruction_text(text);
                        if !cleaned.is_empty() {
                            role_instructions.entry(role).or_default().push(cleaned);
                        }
                    }
                }
            }
            let mut roles: Vec<String> = role_instructions.keys().cloned().collect();
            roles.sort();
            for role in roles {
                if let Some(instructions) = role_instructions.remove(&role) {
                    let instruction = instructions.join("\n");
                    stages.insert(
                        role,
                        StageInfo {
                            instruction,
                            rules: HashMap::new(),
                            temperature: None,
                            prompts: None,
                        },
                    );
                }
            }
        }
    }

    if stages.is_empty() {
        let messages = collect_messages(candidate, &obj);
        if !messages.is_empty() {
            for msg in messages {
                if let Some(map) = msg.as_object() {
                    let content = map
                        .get("content")
                        .or_else(|| map.get("pattern"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if content.is_empty() {
                        continue;
                    }
                    let role = map
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("system")
                        .to_lowercase();
                    stages
                        .entry(role)
                        .and_modify(|stage| {
                            stage.instruction.push_str("\n");
                            stage.instruction.push_str(content);
                        })
                        .or_insert_with(|| StageInfo {
                            instruction: content.to_string(),
                            rules: HashMap::new(),
                            temperature: None,
                            prompts: None,
                        });
                }
            }
        }
    }

    if stages.is_empty() {
        if require_stages {
            let candidate_keys = candidate
                .as_object()
                .map(|map| map.keys().cloned().collect::<Vec<String>>())
                .unwrap_or_default();
            let obj_keys = obj.keys().cloned().collect::<Vec<String>>();
            return Err(CoreError::Validation(format!(
                "Failed to extract stages for candidate {}. candidate keys: {:?}, object keys: {:?}. Expected one of: genome, text_replacements, messages",
                cid, candidate_keys, obj_keys
            )));
        }
        return Ok(None);
    }

    Ok(Some(stages))
}

pub fn extract_program_candidate_content(candidate: &Value) -> String {
    let obj = candidate
        .get("object")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    if let Some(messages) = candidate
        .get("pattern")
        .or_else(|| obj.get("pattern"))
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("messages"))
        .and_then(|v| v.as_array())
    {
        let mut parts = Vec::new();
        for msg in messages.iter().take(5) {
            if let Some(map) = msg.as_object() {
                let role = map
                    .get("role")
                    .or_else(|| map.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("system");
                let content = map
                    .get("pattern")
                    .or_else(|| map.get("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !content.is_empty() {
                    parts.push(format!("[{}]: {}", role.to_uppercase(), content));
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    if let Some(sections) = candidate
        .get("template")
        .or_else(|| obj.get("template"))
        .and_then(|v| v.as_object())
        .and_then(|map| map.get("sections"))
        .and_then(|v| v.as_array())
    {
        let mut parts = Vec::new();
        for section in sections.iter().take(5) {
            if let Some(map) = section.as_object() {
                let role = map
                    .get("role")
                    .or_else(|| map.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("system");
                let content = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !content.is_empty() {
                    parts.push(format!("[{}]: {}", role.to_uppercase(), content));
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    if let Ok(Some(stages)) = extract_stages_from_candidate(candidate, false, None) {
        let mut keys: Vec<String> = stages.keys().cloned().collect();
        keys.sort();
        let mut parts = Vec::new();
        for key in keys {
            if let Some(stage) = stages.get(&key) {
                if !stage.instruction.is_empty() {
                    parts.push(format!("[{}]: {}", key.to_uppercase(), stage.instruction));
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    if let Some(text) = candidate.get("prompt_text").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            return text.to_string();
        }
    }

    let mut parts = Vec::new();
    let replacements = collect_text_replacements(candidate, &obj);
    if !replacements.is_empty() {
        for repl in replacements.iter().take(5) {
            if let Some(map) = repl.as_object() {
                let new_text = map.get("new_text").and_then(|v| v.as_str()).unwrap_or("");
                if new_text.is_empty() {
                    continue;
                }
                let role = map
                    .get("apply_to_role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("system");
                parts.push(format!("[{}]: {}", role.to_uppercase(), new_text));
            }
        }
    }

    if parts.is_empty() {
        let messages = collect_messages(candidate, &obj);
        for msg in messages.iter().take(5) {
            if let Some(map) = msg.as_object() {
                let role = map.get("role").and_then(|v| v.as_str()).unwrap_or("system");
                let content = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !content.is_empty() {
                    parts.push(format!("[{}]: {}", role.to_uppercase(), content));
                }
            }
        }
    }

    if parts.is_empty() {
        let sections = obj
            .get("sections")
            .and_then(|v| v.as_array())
            .or_else(|| {
                obj.get("data")
                    .and_then(|v| v.as_object())
                    .and_then(|map| map.get("sections"))
                    .and_then(|v| v.as_array())
            })
            .or_else(|| candidate.get("sections").and_then(|v| v.as_array()));
        if let Some(sections) = sections {
            for section in sections.iter().take(5) {
                if let Some(map) = section.as_object() {
                    let role = map
                        .get("role")
                        .or_else(|| map.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("system");
                    let content = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if !content.is_empty() {
                        parts.push(format!("[{}]: {}", role.to_uppercase(), content));
                    }
                }
            }
        }
    }

    if parts.is_empty() {
        let instruction = obj
            .get("instruction_text")
            .or_else(|| obj.get("prompt_text"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                candidate
                    .get("transformation")
                    .and_then(|v| v.as_object())
                    .and_then(|map| {
                        map.get("instruction_text")
                            .or_else(|| map.get("prompt_text"))
                    })
                    .and_then(|v| v.as_str())
            });
        if let Some(text) = instruction {
            parts.push(text.to_string());
        }
    }

    parts.join("\n")
}

pub fn normalize_transformation(transformation: &Value) -> Option<Value> {
    if transformation.is_null() {
        return None;
    }

    if let Some(arr) = transformation.as_array() {
        let mut map = Map::new();
        map.insert("text_replacements".to_string(), Value::Array(arr.clone()));
        return Some(Value::Object(map));
    }

    if let Some(obj) = transformation.as_object() {
        let mut map = Map::new();
        if let Some(repl) = obj.get("text_replacements") {
            map.insert("text_replacements".to_string(), repl.clone());
        }
        if let Some(data) = obj.get("data").and_then(|v| v.as_object()) {
            if let Some(repl) = data.get("text_replacements") {
                map.insert("text_replacements".to_string(), repl.clone());
            }
        }
        for key in ["mutation_type", "parent_id", "version_id"] {
            if let Some(val) = obj.get(key) {
                map.insert(key.to_string(), val.clone());
            }
        }
        if map.is_empty() {
            return Some(Value::Object(obj.clone()));
        }
        return Some(Value::Object(map));
    }

    None
}

fn now_timestamp_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn build_program_candidate(
    candidate: &Value,
    candidate_id: Option<&str>,
    seed_info: Option<&Value>,
    token_usage: Option<&Value>,
    cost_usd: Option<f64>,
    timestamp_ms: Option<i64>,
) -> Value {
    let cid = candidate_id
        .map(|s| s.to_string())
        .or_else(|| {
            candidate
                .get("version_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            candidate
                .get("candidate_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let stages = extract_stages_from_candidate(candidate, false, Some(&cid))
        .ok()
        .and_then(|v| v);

    let mut stage_map = Map::new();
    if let Some(stages) = stages {
        for (stage_id, stage) in stages {
            if let Ok(val) = serde_json::to_value(stage) {
                stage_map.insert(stage_id, val);
            }
        }
    } else {
        let prompt_text = extract_program_candidate_content(candidate);
        if !prompt_text.is_empty() {
            let stage = StageInfo {
                instruction: prompt_text,
                rules: HashMap::new(),
                temperature: None,
                prompts: None,
            };
            if let Ok(val) = serde_json::to_value(stage) {
                stage_map.insert("system".to_string(), val);
            }
        } else {
            let stage = StageInfo {
                instruction: String::new(),
                rules: HashMap::new(),
                temperature: None,
                prompts: None,
            };
            if let Ok(val) = serde_json::to_value(stage) {
                stage_map.insert("system".to_string(), val);
            }
        }
    }

    let mut seed_scores = candidate.get("seed_scores").cloned();
    if seed_scores.is_none() {
        let instance_scores = candidate.get("instance_scores").and_then(|v| v.as_array());
        let eval_seeds = candidate
            .get("seed_eval_info")
            .and_then(|v| v.as_object())
            .and_then(|map| map.get("seeds"))
            .and_then(|v| v.as_array());
        if let (Some(scores), Some(seeds)) = (instance_scores, eval_seeds) {
            let count = std::cmp::min(scores.len(), seeds.len());
            let mut entries = Vec::with_capacity(count);
            for idx in 0..count {
                let seed = seeds.get(idx).and_then(|v| value_to_i64(v)).unwrap_or(0);
                let score = scores.get(idx);
                entries.push(seed_reward_entry(seed, score));
            }
            seed_scores = Some(Value::Array(entries));
        }
    }

    let objectives = candidate.get("objectives").cloned().or_else(|| {
        candidate
            .get("score")
            .and_then(|v| v.get("objectives"))
            .cloned()
    });
    let instance_objectives = candidate.get("instance_objectives").cloned().or_else(|| {
        candidate
            .get("score")
            .and_then(|v| v.get("instance_objectives"))
            .cloned()
    });

    let transformation = candidate
        .get("transformation")
        .and_then(|v| normalize_transformation(v));

    let mut map = Map::new();
    map.insert("candidate_id".to_string(), Value::String(cid));
    map.insert(
        "generation".to_string(),
        Value::Number(
            candidate
                .get("generation")
                .and_then(|v| value_to_i64(v))
                .unwrap_or(0)
                .into(),
        ),
    );
    map.insert("stages".to_string(), Value::Object(stage_map));

    if let Some(parent_id) = candidate.get("parent_id").and_then(|v| v.as_str()) {
        map.insert(
            "parent_id".to_string(),
            Value::String(parent_id.to_string()),
        );
    }
    let mutation_type = candidate
        .get("mutation_type")
        .and_then(|v| v.as_str())
        .or_else(|| candidate.get("operator").and_then(|v| v.as_str()))
        .unwrap_or("unknown");
    map.insert(
        "mutation_type".to_string(),
        Value::String(mutation_type.to_string()),
    );

    if let Some(params) = candidate.get("mutation_params") {
        map.insert("mutation_params".to_string(), params.clone());
    }

    let reward = candidate
        .get("accuracy")
        .and_then(|v| value_to_f64(v))
        .unwrap_or(0.0);
    map.insert(
        "reward".to_string(),
        serde_json::Number::from_f64(reward)
            .map(Value::Number)
            .unwrap_or(Value::Null),
    );

    if let Some(val) = candidate.get("val_accuracy").and_then(|v| value_to_f64(v)) {
        if let Some(num) = serde_json::Number::from_f64(val) {
            map.insert("val_reward".to_string(), Value::Number(num));
        }
    } else if let Some(val) = candidate.get("full_score").and_then(|v| value_to_f64(v)) {
        if let Some(num) = serde_json::Number::from_f64(val) {
            map.insert("val_reward".to_string(), Value::Number(num));
        }
    }

    if let Some(val) = candidate
        .get("minibatch_score")
        .and_then(|v| value_to_f64(v))
    {
        if let Some(num) = serde_json::Number::from_f64(val) {
            map.insert("minibatch_reward".to_string(), Value::Number(num));
        }
    }

    if let Some(seed_scores) = seed_scores {
        map.insert("seed_rewards".to_string(), seed_scores);
    }
    if let Some(seed_info) = seed_info {
        if let Ok(parsed) = serde_json::from_value::<Vec<SeedInfo>>(seed_info.clone()) {
            if let Ok(val) = serde_json::to_value(parsed) {
                map.insert("seed_info".to_string(), val);
            }
        } else {
            map.insert("seed_info".to_string(), seed_info.clone());
        }
    }

    if let Some(instance_scores) = candidate.get("instance_scores") {
        map.insert("instance_rewards".to_string(), instance_scores.clone());
    }
    if let Some(objectives) = objectives {
        map.insert("objectives".to_string(), objectives);
    }
    if let Some(instance_objectives) = instance_objectives {
        map.insert("instance_objectives".to_string(), instance_objectives);
    }
    if let Some(value) = candidate.get("newly_solved_seeds") {
        map.insert("newly_solved_seeds".to_string(), value.clone());
    }
    if let Some(value) = candidate.get("artifact_refs") {
        map.insert("artifact_refs".to_string(), value.clone());
    }
    if let Some(value) = candidate.get("success_statuses") {
        map.insert("success_statuses".to_string(), value.clone());
    }

    if let Some(token_usage) = token_usage {
        if let Ok(parsed) = serde_json::from_value::<TokenUsage>(token_usage.clone()) {
            if let Ok(val) = serde_json::to_value(parsed) {
                map.insert("token_usage".to_string(), val);
            }
        } else {
            map.insert("token_usage".to_string(), token_usage.clone());
        }
    }
    if let Some(cost) = cost_usd {
        if let Some(num) = serde_json::Number::from_f64(cost) {
            map.insert("cost_usd".to_string(), Value::Number(num));
        }
    }

    map.insert(
        "timestamp_ms".to_string(),
        Value::Number(timestamp_ms.unwrap_or_else(now_timestamp_ms).into()),
    );

    if let Some(val) = candidate.get("evaluation_duration_ms") {
        map.insert("evaluation_duration_ms".to_string(), val.clone());
    }

    if let Some(transformation) = transformation {
        map.insert("transformation".to_string(), transformation);
    }
    if let Some(value) = candidate.get("prompt_length") {
        map.insert("prompt_length".to_string(), value.clone());
    }
    let status = candidate
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("evaluated");
    map.insert("status".to_string(), Value::String(status.to_string()));

    for key in [
        "context_override_bundle_id",
        "context_overrides",
        "override_application_status",
        "override_application_errors",
        "context_snapshot_ref",
    ] {
        if let Some(val) = candidate.get(key) {
            map.insert(key.to_string(), val.clone());
        }
    }

    Value::Object(map)
}
