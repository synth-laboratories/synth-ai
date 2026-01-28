//! GraphGen (graph optimization) validation helpers.

use serde_json::{Map, Value};
use std::collections::HashSet;
use crate::errors::CoreError;

#[derive(Debug, Clone, Default)]
pub struct GraphGenValidationResult {
    pub errors: Vec<Value>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct GraphOptModels {
    policy_models: HashSet<String>,
}

fn load_graph_opt_models() -> Option<GraphOptModels> {
    let raw = include_str!("../../assets/supported_models.json");
    let value: Value = serde_json::from_str(raw).ok()?;
    let graph_opt = value.get("graph_opt")?.as_object()?;
    let policy_models = graph_opt.get("policy_models")?.as_array()?;
    let mut set = HashSet::new();
    for item in policy_models {
        if let Some(name) = item.as_str() {
            set.insert(name.to_string());
        }
    }
    Some(GraphOptModels { policy_models: set })
}

static GRAPH_OPT_MODELS: once_cell::sync::Lazy<Option<GraphOptModels>> =
    once_cell::sync::Lazy::new(load_graph_opt_models);

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn parse_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len()).max(1);
    let dist = levenshtein(&a_chars, &b_chars) as f64;
    1.0 - (dist / max_len as f64)
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut costs: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = costs[0];
        costs[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let temp = costs[j + 1];
            let mut new_cost = prev + if ca == cb { 0 } else { 1 };
            new_cost = new_cost.min(costs[j] + 1).min(temp + 1);
            costs[j + 1] = new_cost;
            prev = temp;
        }
    }
    costs[b.len()]
}

fn find_similar_models(model: &str, supported: &HashSet<String>) -> Vec<String> {
    let mut scored: Vec<(f64, String)> = supported
        .iter()
        .map(|candidate| (similarity(model, candidate), candidate.clone()))
        .filter(|(score, _)| *score >= 0.4)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(3).map(|(_, m)| m).collect()
}

fn push_error(
    errors: &mut Vec<Value>,
    field: &str,
    error: String,
    suggestion: Option<String>,
    similar: Option<Vec<String>>,
) {
    let mut map = Map::new();
    map.insert("field".to_string(), Value::String(field.to_string()));
    map.insert("error".to_string(), Value::String(error));
    if let Some(suggestion) = suggestion {
        map.insert("suggestion".to_string(), Value::String(suggestion));
    }
    if let Some(similar) = similar {
        map.insert(
            "similar".to_string(),
            Value::Array(similar.into_iter().map(Value::String).collect()),
        );
    }
    errors.push(Value::Object(map));
}

pub fn validate_graphgen_job_config(
    config: &Value,
    dataset: &Value,
) -> GraphGenValidationResult {
    let mut result = GraphGenValidationResult::default();

    let config_map = match config.as_object() {
        Some(map) => map,
        None => {
            push_error(
                &mut result.errors,
                "policy_models",
                "policy_models is required".to_string(),
                None,
                None,
            );
            return result;
        }
    };

    let policy_models_raw = config_map
        .get("policy_models")
        .or_else(|| config_map.get("policy_model"))
        .or_else(|| config_map.get("model"));
    let mut policy_models_list: Vec<String> = Vec::new();
    match policy_models_raw {
        None => {
            push_error(
                &mut result.errors,
                "policy_models",
                "policy_models is required".to_string(),
                Some("Supported models: see graph_opt.policy_models".to_string()),
                None,
            );
        }
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(name) = value_to_string(item) {
                    policy_models_list.push(name);
                } else {
                    push_error(
                        &mut result.errors,
                        "policy_models",
                        "policy_models contains empty value".to_string(),
                        Some("Supported models: see graph_opt.policy_models".to_string()),
                        None,
                    );
                }
            }
        }
        Some(value) => {
            if let Some(name) = value_to_string(value) {
                policy_models_list.push(name);
            } else {
                push_error(
                    &mut result.errors,
                    "policy_models",
                    "policy_models contains empty value".to_string(),
                    Some("Supported models: see graph_opt.policy_models".to_string()),
                    None,
                );
            }
        }
    }

    if let Some(models) = GRAPH_OPT_MODELS.as_ref() {
        for policy_model in &policy_models_list {
            let clean = policy_model.trim();
            if clean.is_empty() {
                push_error(
                    &mut result.errors,
                    "policy_models",
                    "policy_models contains empty value".to_string(),
                    Some("Supported models: see graph_opt.policy_models".to_string()),
                    None,
                );
                continue;
            }
            if !models.policy_models.contains(clean) {
                let similar = find_similar_models(clean, &models.policy_models);
                push_error(
                    &mut result.errors,
                    "policy_models",
                    format!("Unsupported policy model: {}", clean),
                    Some(format!("Supported models: {:?}", models.policy_models)),
                    if similar.is_empty() { None } else { Some(similar) },
                );
            }
        }
    }

    if let Some(effort) = config_map.get("proposer_effort").and_then(|v| v.as_str()) {
        if effort != "low" && effort != "medium" && effort != "high" {
            push_error(
                &mut result.errors,
                "proposer_effort",
                format!("Invalid proposer_effort: {}", effort),
                Some("Must be one of: 'low', 'medium', 'high'".to_string()),
                None,
            );
        }
    }

    let rollout_budget = config_map
        .get("rollout_budget")
        .or_else(|| config_map.get("budget"))
        .and_then(parse_int)
        .unwrap_or(100);
    if rollout_budget < 10 {
        push_error(
            &mut result.errors,
            "rollout_budget",
            format!("rollout_budget must be >= 10, got {}", rollout_budget),
            None,
            None,
        );
    }
    if rollout_budget > 10000 {
        push_error(
            &mut result.errors,
            "rollout_budget",
            format!("rollout_budget must be <= 10000, got {}", rollout_budget),
            None,
            None,
        );
    }

    let dataset_map = match dataset.as_object() {
        Some(map) => map,
        None => {
            push_error(
                &mut result.errors,
                "dataset",
                "dataset must be a dict".to_string(),
                None,
                None,
            );
            return result;
        }
    };

    match dataset_map.get("tasks") {
        Some(Value::Array(tasks)) => {
            if tasks.is_empty() {
                push_error(
                    &mut result.errors,
                    "dataset.tasks",
                    "Dataset must contain at least one task".to_string(),
                    None,
                    None,
                );
            } else if tasks.len() < 2 {
                result.warnings.push(
                    "GraphGen datasets with <2 tasks are unlikely to optimize meaningfully."
                        .to_string(),
                );
            }
        }
        _ => {
            push_error(
                &mut result.errors,
                "dataset.tasks",
                "Dataset must contain at least one task".to_string(),
                None,
                None,
            );
        }
    }

    result
}

/// Return the graph optimization supported models config from assets.
pub fn graph_opt_supported_models() -> Value {
    let raw = include_str!("../../assets/supported_models.json");
    let value: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Value::Object(Map::new()),
    };
    value
        .get("graph_opt")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()))
}

/// Validate a GraphGen taskset payload (dataset-only validation).
pub fn validate_graphgen_taskset(dataset: &Value) -> Vec<Value> {
    let mut errors: Vec<Value> = Vec::new();

    let dataset_map = match dataset.as_object() {
        Some(map) => map,
        None => {
            push_error(
                &mut errors,
                "dataset",
                "dataset must be an object".to_string(),
                None,
                None,
            );
            return errors;
        }
    };

    // Metadata name
    match dataset_map.get("metadata") {
        Some(Value::Object(meta)) => {
            match meta.get("name") {
                Some(Value::String(name)) if !name.trim().is_empty() => {}
                _ => {
                    push_error(
                        &mut errors,
                        "metadata.name",
                        "metadata.name is required".to_string(),
                        None,
                        None,
                    );
                }
            }
        }
        _ => {
            push_error(
                &mut errors,
                "metadata",
                "metadata is required".to_string(),
                None,
                None,
            );
        }
    }

    // Tasks
    let mut task_ids: HashSet<String> = HashSet::new();
    match dataset_map.get("tasks") {
        Some(Value::Array(tasks)) => {
            if tasks.is_empty() {
                push_error(
                    &mut errors,
                    "tasks",
                    "dataset must contain at least one task".to_string(),
                    None,
                    None,
                );
            }
            for (idx, task) in tasks.iter().enumerate() {
                match task.as_object() {
                    Some(task_map) => match task_map.get("id") {
                        Some(Value::String(id)) if !id.trim().is_empty() => {
                            if !task_ids.insert(id.to_string()) {
                                push_error(
                                    &mut errors,
                                    "tasks.id",
                                    format!("duplicate task id '{}'", id),
                                    None,
                                    None,
                                );
                            }
                        }
                        _ => {
                            push_error(
                                &mut errors,
                                &format!("tasks[{}].id", idx),
                                "task id is required".to_string(),
                                None,
                                None,
                            );
                        }
                    },
                    None => {
                        push_error(
                            &mut errors,
                            &format!("tasks[{}]", idx),
                            "task must be an object".to_string(),
                            None,
                            None,
                        );
                    }
                }
            }
        }
        _ => {
            push_error(
                &mut errors,
                "tasks",
                "dataset.tasks must be a non-empty list".to_string(),
                None,
                None,
            );
        }
    }

    // Gold outputs must reference valid tasks if task_id provided.
    if let Some(Value::Array(gold_outputs)) = dataset_map.get("gold_outputs") {
        for (idx, gold) in gold_outputs.iter().enumerate() {
            if let Some(gold_map) = gold.as_object() {
                if let Some(Value::String(task_id)) = gold_map.get("task_id") {
                    if !task_id.is_empty() && !task_ids.contains(task_id) {
                        push_error(
                            &mut errors,
                            &format!("gold_outputs[{}].task_id", idx),
                            format!("invalid task_id '{}'", task_id),
                            None,
                            None,
                        );
                    }
                }
            }
        }
    }

    // select_output validation
    let select_output = dataset_map
        .get("select_output")
        .or_else(|| {
            dataset_map
                .get("metadata")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get("select_output"))
        });
    if let Some(value) = select_output {
        match value {
            Value::String(_) => {}
            Value::Array(items) => {
                if !items.iter().all(|item| item.as_str().is_some()) {
                    push_error(
                        &mut errors,
                        "select_output",
                        "select_output must be a string or list of strings".to_string(),
                        None,
                        None,
                    );
                }
            }
            _ => {
                push_error(
                    &mut errors,
                    "select_output",
                    "select_output must be a string or list of strings".to_string(),
                    None,
                    None,
                );
            }
        }
    }

    // output_config validation
    let output_config = dataset_map
        .get("output_config")
        .or_else(|| {
            dataset_map
                .get("metadata")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get("output_config"))
        });
    if let Some(value) = output_config {
        if !value.is_object() {
            push_error(
                &mut errors,
                "output_config",
                "output_config must be an object".to_string(),
                None,
                None,
            );
        }
    }

    // input_schema/output_schema type checks
    for field in ["input_schema", "output_schema"] {
        let value = dataset_map.get(field).or_else(|| {
            dataset_map
                .get("metadata")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get(field))
        });
        if let Some(v) = value {
            if !v.is_object() {
                push_error(
                    &mut errors,
                    field,
                    format!("{} must be an object", field),
                    None,
                    None,
                );
            }
        }
    }

    errors
}

/// Parse and validate a GraphGen taskset payload.
pub fn parse_graphgen_taskset(dataset: &Value) -> Result<Value, CoreError> {
    let errors = validate_graphgen_taskset(dataset);
    if !errors.is_empty() {
        return Err(CoreError::Validation(format!(
            "invalid GraphGenTaskSet: {} errors",
            errors.len()
        )));
    }
    Ok(dataset.clone())
}

/// Load and validate a GraphGen taskset from a JSON file.
pub fn load_graphgen_taskset(path: &std::path::Path) -> Result<Value, CoreError> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        CoreError::InvalidInput(format!("failed to read dataset file '{}': {}", path.display(), e))
    })?;
    let value: Value = serde_json::from_str(&contents).map_err(|e| {
        CoreError::Validation(format!(
            "failed to parse dataset JSON '{}': {}",
            path.display(),
            e
        ))
    })?;
    parse_graphgen_taskset(&value)
}

// =============================================================================
// Graph TOML validation helpers
// =============================================================================

fn push_graph_error(
    errors: &mut Vec<Value>,
    field: &str,
    error: String,
    suggestion: Option<String>,
) {
    let mut map = Map::new();
    map.insert("field".to_string(), Value::String(field.to_string()));
    map.insert("error".to_string(), Value::String(error));
    if let Some(suggestion) = suggestion {
        map.insert("suggestion".to_string(), Value::String(suggestion));
    }
    errors.push(Value::Object(map));
}

fn value_to_bool(value: &Value, default_value: bool) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Number(n) => n.as_i64().map(|v| v != 0).unwrap_or(default_value),
        Value::String(s) => {
            let trimmed = s.trim().to_lowercase();
            match trimmed.as_str() {
                "true" | "1" | "yes" => true,
                "false" | "0" | "no" => false,
                _ => default_value,
            }
        }
        _ => default_value,
    }
}

fn normalize_policy_models(raw: Option<&Value>, errors: &mut Vec<Value>) -> Vec<String> {
    match raw {
        None => {
            push_graph_error(
                errors,
                "policy_models",
                "policy_models is required".to_string(),
                None,
            );
            Vec::new()
        }
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(value_to_string)
            .collect::<Vec<String>>(),
        Some(value) => value_to_string(value).map(|v| vec![v]).unwrap_or_default(),
    }
}

fn build_graph_config(
    section: &Map<String, Value>,
    errors: &mut Vec<Value>,
) -> Value {
    let policy_models_raw = section
        .get("policy_models")
        .or_else(|| section.get("policy_model"))
        .or_else(|| section.get("model"));
    let policy_models = normalize_policy_models(policy_models_raw, errors);

    let rollout_budget = section
        .get("rollout_budget")
        .or_else(|| section.get("budget"))
        .and_then(parse_int)
        .unwrap_or(100);

    let proposer_effort = section
        .get("proposer_effort")
        .or_else(|| section.get("effort"))
        .and_then(|v| v.as_str())
        .unwrap_or("medium")
        .to_string();

    let mut map = Map::new();
    map.insert(
        "policy_models".to_string(),
        Value::Array(policy_models.into_iter().map(Value::String).collect()),
    );
    if let Some(provider) = section.get("policy_provider").and_then(|v| v.as_str()) {
        map.insert("policy_provider".to_string(), Value::String(provider.to_string()));
    }
    map.insert("rollout_budget".to_string(), Value::Number(rollout_budget.into()));
    map.insert(
        "proposer_effort".to_string(),
        Value::String(proposer_effort),
    );
    if let Some(verifier_model) = section.get("verifier_model").and_then(|v| v.as_str()) {
        map.insert(
            "verifier_model".to_string(),
            Value::String(verifier_model.to_string()),
        );
    }
    if let Some(verifier_provider) = section.get("verifier_provider").and_then(|v| v.as_str()) {
        map.insert(
            "verifier_provider".to_string(),
            Value::String(verifier_provider.to_string()),
        );
    }
    if let Some(population_size) = section.get("population_size").and_then(parse_int) {
        map.insert(
            "population_size".to_string(),
            Value::Number(population_size.into()),
        );
    } else {
        map.insert("population_size".to_string(), Value::Number(4.into()));
    }
    if let Some(num_generations) = section.get("num_generations").and_then(parse_int) {
        map.insert(
            "num_generations".to_string(),
            Value::Number(num_generations.into()),
        );
    }

    Value::Object(map)
}

pub fn validate_graph_job_section(
    section: &Value,
    base_dir: Option<&std::path::Path>,
) -> (Option<Value>, Vec<Value>) {
    let mut errors: Vec<Value> = Vec::new();
    let section_map = match section.as_object() {
        Some(map) => map,
        None => {
            push_graph_error(
                &mut errors,
                "graph",
                "graph section must be a table".to_string(),
                None,
            );
            return (None, errors);
        }
    };

    let dataset_ref = section_map
        .get("dataset_path")
        .or_else(|| section_map.get("dataset"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let (dataset_path, dataset_value) = if let Some(path_str) = dataset_ref {
        let mut path = std::path::PathBuf::from(path_str.clone());
        if let Some(base) = base_dir {
            if path.is_relative() {
                path = base.join(path);
            }
        }
        let resolved = path.clone();
        let data = std::fs::read_to_string(&resolved);
        match data {
            Ok(contents) => match serde_json::from_str::<Value>(&contents) {
                Ok(value) => (Some(resolved), Some(value)),
                Err(err) => {
                    push_graph_error(
                        &mut errors,
                        "graph.dataset",
                        format!("Invalid GraphGenTaskSet JSON: {}", err),
                        None,
                    );
                    (Some(resolved), None)
                }
            },
            Err(_) => {
                push_graph_error(
                    &mut errors,
                    "graph.dataset",
                    format!("Dataset file not found: {}", resolved.display()),
                    None,
                );
                (Some(resolved), None)
            }
        }
    } else {
        push_graph_error(
            &mut errors,
            "graph.dataset",
            "dataset (path) is required".to_string(),
            Some("Set graph.dataset = \"my_tasks.json\"".to_string()),
        );
        (None, None)
    };

    let config_value = build_graph_config(section_map, &mut errors);

    let auto_start = section_map
        .get("auto_start")
        .map(|v| value_to_bool(v, true))
        .unwrap_or(true);
    let metadata = match section_map.get("metadata") {
        Some(Value::Object(map)) => Value::Object(map.clone()),
        _ => Value::Object(Map::new()),
    };
    let initial_prompt = section_map
        .get("initial_prompt")
        .and_then(|v| v.as_str())
        .map(|s| Value::String(s.to_string()));

    if let Some(dataset) = dataset_value.as_ref() {
        let validation = validate_graphgen_job_config(&config_value, dataset);
        errors.extend(validation.errors);
    }

    if !errors.is_empty() {
        return (None, errors);
    }

    let mut result = Map::new();
    if let Some(path) = dataset_path {
        result.insert(
            "dataset_path".to_string(),
            Value::String(path.to_string_lossy().to_string()),
        );
    }
    if let Some(dataset) = dataset_value {
        result.insert("dataset".to_string(), dataset);
    }
    result.insert("config".to_string(), config_value);
    result.insert("auto_start".to_string(), Value::Bool(auto_start));
    result.insert("metadata".to_string(), metadata);
    if let Some(prompt) = initial_prompt {
        result.insert("initial_prompt".to_string(), prompt);
    }

    (Some(Value::Object(result)), errors)
}

pub fn load_graph_job_toml(path: &std::path::Path) -> (Option<Value>, Vec<Value>) {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            let mut errors = Vec::new();
            push_graph_error(
                &mut errors,
                "graph",
                format!("Failed to read TOML: {}", err),
                None,
            );
            return (None, errors);
        }
    };

    let toml_value: Value = match crate::config::parse_toml(&content) {
        Ok(value) => value,
        Err(err) => {
            let mut errors = Vec::new();
            push_graph_error(
                &mut errors,
                "graph",
                format!("Failed to parse TOML: {}", err),
                None,
            );
            return (None, errors);
        }
    };

    let graph_section = toml_value
        .get("graph")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));

    validate_graph_job_section(&graph_section, path.parent())
}

pub fn validate_graph_job_payload(payload: &Value) -> Vec<Value> {
    let mut errors = Vec::new();
    let payload_map = match payload.as_object() {
        Some(map) => map,
        None => {
            push_graph_error(
                &mut errors,
                "payload",
                "Payload must be an object".to_string(),
                None,
            );
            return errors;
        }
    };

    let dataset = match payload_map.get("dataset") {
        Some(Value::Object(map)) => Value::Object(map.clone()),
        Some(_) => {
            push_graph_error(
                &mut errors,
                "dataset",
                "dataset must be a dict".to_string(),
                None,
            );
            return errors;
        }
        None => {
            push_graph_error(
                &mut errors,
                "dataset",
                "dataset must be a dict".to_string(),
                None,
            );
            return errors;
        }
    };

    let metadata: Map<String, Value> = match payload_map.get("metadata") {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    };

    let policy_models_raw = payload_map
        .get("policy_models")
        .or_else(|| payload_map.get("policy_model"));
    let policy_models = normalize_policy_models(policy_models_raw, &mut errors);

    let rollout_budget = payload_map
        .get("rollout_budget")
        .and_then(parse_int)
        .unwrap_or(100);
    let proposer_effort = payload_map
        .get("proposer_effort")
        .and_then(|v| v.as_str())
        .unwrap_or("medium")
        .to_string();

    let mut config_map = Map::new();
    config_map.insert(
        "policy_models".to_string(),
        Value::Array(policy_models.into_iter().map(Value::String).collect()),
    );
    if let Some(policy_provider) = payload_map.get("policy_provider").and_then(|v| v.as_str()) {
        config_map.insert(
            "policy_provider".to_string(),
            Value::String(policy_provider.to_string()),
        );
    }
    config_map.insert("rollout_budget".to_string(), Value::Number(rollout_budget.into()));
    config_map.insert(
        "proposer_effort".to_string(),
        Value::String(proposer_effort),
    );
    if let Some(verifier_model) = payload_map.get("verifier_model").and_then(|v| v.as_str()) {
        config_map.insert(
            "verifier_model".to_string(),
            Value::String(verifier_model.to_string()),
        );
    }
    if let Some(verifier_provider) = payload_map
        .get("verifier_provider")
        .and_then(|v| v.as_str())
    {
        config_map.insert(
            "verifier_provider".to_string(),
            Value::String(verifier_provider.to_string()),
        );
    }
    if let Some(population_size) = metadata.get("population_size").and_then(parse_int) {
        config_map.insert(
            "population_size".to_string(),
            Value::Number(population_size.into()),
        );
    }
    if let Some(num_generations) = metadata.get("num_generations").and_then(parse_int) {
        config_map.insert(
            "num_generations".to_string(),
            Value::Number(num_generations.into()),
        );
    }

    let config_value = Value::Object(config_map);
    let validation = validate_graphgen_job_config(&config_value, &dataset);
    errors.extend(validation.errors);
    errors
}
