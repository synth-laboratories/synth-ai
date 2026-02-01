//! Configuration utilities for Synth SDK.
//!
//! This module provides:
//! - Core SDK configuration (CoreConfig)
//! - TOML file parsing
//! - Config deep merge
//! - Optimization defaults

use crate::errors::CoreError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::path::Path;

/// Backend auth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendAuth {
    /// Use `X-API-Key` header.
    XApiKey,
    /// Use `Authorization: Bearer` header.
    Bearer,
}

impl Default for BackendAuth {
    fn default() -> Self {
        BackendAuth::XApiKey
    }
}

/// Core config shared across urls/events/tunnels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub backend_base_url: String,
    pub api_key: Option<String>,
    pub user_agent: String,
    pub timeout_ms: u64,
    pub retries: u32,
    pub auth: BackendAuth,
}

impl Default for CoreConfig {
    fn default() -> Self {
        let backend_base_url = std::env::var("SYNTH_BACKEND_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "https://api.usesynth.ai".to_string());
        let api_key = std::env::var("SYNTH_API_KEY").ok();
        CoreConfig {
            backend_base_url,
            api_key,
            user_agent: "synth-core/0.1".to_string(),
            timeout_ms: 30_000,
            retries: 3,
            auth: BackendAuth::default(),
        }
    }
}

impl CoreConfig {
    pub fn with_backend(mut self, backend_base_url: String) -> Self {
        self.backend_base_url = backend_base_url;
        self
    }
}

// =============================================================================
// TOML Parsing
// =============================================================================

/// Load a TOML file and convert to JSON Value.
///
/// # Arguments
///
/// * `path` - Path to the TOML file
pub fn load_toml(path: &Path) -> Result<Value, CoreError> {
    let content = fs::read_to_string(path)
        .map_err(|e| CoreError::Config(format!("failed to read TOML file: {}", e)))?;

    parse_toml(&content)
}

/// Parse a TOML string to JSON Value.
pub fn parse_toml(content: &str) -> Result<Value, CoreError> {
    let toml_value: toml::Value = toml::from_str(content)
        .map_err(|e| CoreError::Config(format!("failed to parse TOML: {}", e)))?;

    // Convert TOML to JSON
    toml_to_json(toml_value)
}

/// Convert a TOML Value to JSON Value.
fn toml_to_json(toml: toml::Value) -> Result<Value, CoreError> {
    match toml {
        toml::Value::String(s) => Ok(Value::String(s)),
        toml::Value::Integer(i) => Ok(Value::Number(i.into())),
        toml::Value::Float(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Config("invalid float value".to_string())),
        toml::Value::Boolean(b) => Ok(Value::Bool(b)),
        toml::Value::Datetime(dt) => Ok(Value::String(dt.to_string())),
        toml::Value::Array(arr) => {
            let json_arr: Result<Vec<Value>, CoreError> =
                arr.into_iter().map(toml_to_json).collect();
            Ok(Value::Array(json_arr?))
        }
        toml::Value::Table(table) => {
            let mut map = serde_json::Map::new();
            for (k, v) in table {
                map.insert(k, toml_to_json(v)?);
            }
            Ok(Value::Object(map))
        }
    }
}

// =============================================================================
// Deep Merge
// =============================================================================

/// Deep merge two JSON values.
///
/// For objects, keys from `overrides` replace or add to `base`.
/// For other types, `overrides` completely replaces `base`.
///
/// # Arguments
///
/// * `base` - Base value to merge into (modified in place)
/// * `overrides` - Override values to apply
pub fn deep_merge(base: &mut Value, overrides: &Value) {
    match (base, overrides) {
        (Value::Object(base_map), Value::Object(override_map)) => {
            for (key, override_val) in override_map {
                if let Some(base_val) = base_map.get_mut(key) {
                    deep_merge(base_val, override_val);
                } else {
                    base_map.insert(key.clone(), override_val.clone());
                }
            }
        }
        (base, overrides) => {
            *base = overrides.clone();
        }
    }
}

/// Deep update a JSON value with support for dot-notation keys.
///
/// This mirrors the Python `deep_update` helper, allowing overrides like
/// "prompt_learning.gepa.rollout.budget" to create nested objects.
pub fn deep_update(base: &mut Value, overrides: &Value) {
    if !overrides.is_object() {
        return;
    }
    if !base.is_object() {
        *base = Value::Object(Map::new());
    }

    let base_map = base.as_object_mut().expect("base is object");
    let override_map = overrides.as_object().expect("overrides is object");

    for (key, override_val) in override_map {
        if key.contains('.') {
            apply_dot_update(base_map, key, override_val);
        } else {
            match base_map.get_mut(key) {
                Some(existing) => {
                    if existing.is_object() && override_val.is_object() {
                        deep_update(existing, override_val);
                    } else {
                        *existing = override_val.clone();
                    }
                }
                None => {
                    base_map.insert(key.clone(), override_val.clone());
                }
            }
        }
    }
}

fn apply_dot_update(base_map: &mut Map<String, Value>, key: &str, override_val: &Value) {
    let mut current = base_map;
    let mut parts = key.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            match current.get_mut(part) {
                Some(existing) => {
                    if existing.is_object() && override_val.is_object() {
                        deep_update(existing, override_val);
                    } else {
                        *existing = override_val.clone();
                    }
                }
                None => {
                    current.insert(part.to_string(), override_val.clone());
                }
            }
        } else {
            let entry = current
                .entry(part.to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if !entry.is_object() {
                *entry = Value::Object(Map::new());
            }
            current = entry.as_object_mut().expect("entry is object");
        }
    }
}

/// Validate that all override keys exist in the base config.
///
/// This helps catch typos in override configs.
///
/// # Arguments
///
/// * `base` - Base config to validate against
/// * `overrides` - Override config to validate
/// * `path` - Current path for error messages
pub fn validate_overrides(base: &Value, overrides: &Value, path: &str) -> Result<(), CoreError> {
    match (base, overrides) {
        (Value::Object(base_map), Value::Object(override_map)) => {
            for (key, override_val) in override_map {
                let key_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };

                if let Some(base_val) = base_map.get(key) {
                    validate_overrides(base_val, override_val, &key_path)?;
                } else {
                    return Err(CoreError::Config(format!(
                        "unknown config key: {}",
                        key_path
                    )));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// =============================================================================
// Config Value Resolution
// =============================================================================

#[derive(Debug, Clone)]
pub struct ResolvedConfigValue {
    pub value: Option<String>,
    pub cli_value: Option<String>,
    pub config_value: Option<String>,
    pub cli_overrides_config: bool,
}

fn clean_opt(value: Option<&str>) -> Option<String> {
    match value {
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        None => None,
    }
}

/// Resolve a configuration value with CLI > ENV > CONFIG > DEFAULT precedence.
pub fn resolve_config_value(
    cli_value: Option<&str>,
    env_value: Option<&str>,
    config_value: Option<&str>,
    default_value: Option<&str>,
) -> ResolvedConfigValue {
    let cli_clean = clean_opt(cli_value);
    let env_clean = clean_opt(env_value);
    let config_clean = clean_opt(config_value);
    let default_clean = clean_opt(default_value);

    let cli_overrides_config = match (&cli_clean, &config_clean) {
        (Some(cli), Some(config)) => cli != config,
        _ => false,
    };

    let resolved = cli_clean
        .clone()
        .or(env_clean)
        .or(config_clean.clone())
        .or(default_clean);

    ResolvedConfigValue {
        value: resolved,
        cli_value: cli_clean,
        config_value: config_clean,
        cli_overrides_config,
    }
}

// =============================================================================
// Optimization Defaults
// =============================================================================

/// Default values for optimization jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationDefaults {
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Mutation rate (0.0 to 1.0)
    pub mutation_rate: f64,
    /// Crossover rate (0.0 to 1.0)
    pub crossover_rate: f64,
    /// Number of elite individuals to preserve
    pub elite_count: usize,
    /// Training set ratio (0.0 to 1.0)
    pub train_ratio: f64,
    /// Maximum rollouts per candidate
    pub max_rollouts: usize,
    /// Default timeout per rollout in seconds
    pub rollout_timeout_secs: u64,
    /// Enable caching of rollout results
    pub enable_caching: bool,
}

impl Default for OptimizationDefaults {
    fn default() -> Self {
        Self {
            population_size: 10,
            generations: 5,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            elite_count: 2,
            train_ratio: 0.7,
            max_rollouts: 100,
            rollout_timeout_secs: 60,
            enable_caching: true,
        }
    }
}

// =============================================================================
// Prompt-learning config expansion defaults (v1)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionDefaultsV1 {
    pub version: String,
    pub train_ratio: f64,
    pub rollout_budget: i64,
    pub rollout_max_concurrent: i64,
    pub mutation_rate: f64,
    pub pop_size_min: i64,
    pub pop_size_max: i64,
    pub pop_size_divisor: i64,
    pub num_generations: i64,
    pub children_divisor: i64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub archive_multiplier: i64,
    pub pareto_eps: f64,
    pub feedback_fraction: f64,
    pub eval_max_concurrent: i64,
    pub eval_timeout: f64,
}

impl ExpansionDefaultsV1 {
    pub fn v1() -> Self {
        Self {
            version: "v1".to_string(),
            train_ratio: 0.7,
            rollout_budget: 100_000_000,
            rollout_max_concurrent: 20,
            mutation_rate: 0.3,
            pop_size_min: 10,
            pop_size_max: 30,
            pop_size_divisor: 10,
            num_generations: 10,
            children_divisor: 4,
            crossover_rate: 0.5,
            selection_pressure: 1.0,
            archive_multiplier: 2,
            pareto_eps: 1e-6,
            feedback_fraction: 0.5,
            eval_max_concurrent: 20,
            eval_timeout: 600.0,
        }
    }
}

pub fn expansion_defaults(version: Option<&str>) -> Result<ExpansionDefaultsV1, CoreError> {
    match version.unwrap_or("v1") {
        "v1" => Ok(ExpansionDefaultsV1::v1()),
        other => Err(CoreError::Config(format!(
            "unknown defaults version: {}",
            other
        ))),
    }
}

// =============================================================================
// Seed Resolution
// =============================================================================

/// Resolve seeds from various input formats.
///
/// Handles:
/// - Array of strings: ["seed1", "seed2"]
/// - Object with ids: {"seed1": {...}, "seed2": {...}}
/// - Single string: "seed1"
///
/// # Arguments
///
/// * `seeds` - Seed value in any supported format
pub fn resolve_seeds(seeds: &Value) -> Result<Vec<String>, CoreError> {
    match seeds {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for item in arr {
                match item {
                    Value::String(s) => result.push(s.clone()),
                    Value::Object(obj) => {
                        if let Some(Value::String(id)) = obj.get("id") {
                            result.push(id.clone());
                        }
                    }
                    _ => {}
                }
            }
            Ok(result)
        }
        Value::Object(obj) => Ok(obj.keys().cloned().collect()),
        Value::String(s) => Ok(vec![s.clone()]),
        _ => Ok(Vec::new()),
    }
}

/// Split seeds into training and validation sets.
///
/// # Arguments
///
/// * `seeds` - List of seed IDs
/// * `train_ratio` - Ratio of seeds for training (e.g., 0.7 for 70%)
pub fn split_train_validation(seeds: &[String], train_ratio: f64) -> (Vec<String>, Vec<String>) {
    let ratio = train_ratio.clamp(0.0, 1.0);
    let train_count = ((seeds.len() as f64) * ratio).round() as usize;
    let train_count = train_count.max(1).min(seeds.len());

    let train = seeds[..train_count].to_vec();
    let validation = seeds[train_count..].to_vec();

    (train, validation)
}

// =============================================================================
// Config Expansion
// =============================================================================

/// Expand a minimal config with defaults.
///
/// # Arguments
///
/// * `minimal` - Minimal config with user-specified values
/// * `defaults` - Default values to fill in
pub fn expand_config(minimal: &Value, defaults: &OptimizationDefaults) -> Result<Value, CoreError> {
    let defaults_json = serde_json::to_value(defaults)
        .map_err(|e| CoreError::Config(format!("failed to serialize defaults: {}", e)))?;

    let mut expanded = defaults_json;
    deep_merge(&mut expanded, minimal);

    // Handle population_size based on seed count if not specified
    if minimal.get("population_size").is_none() {
        if let Some(seeds) = minimal.get("seeds") {
            let seed_count = resolve_seeds(seeds)?.len();
            if let Value::Object(ref mut map) = expanded {
                map.insert(
                    "population_size".to_string(),
                    Value::Number((seed_count.max(10)).into()),
                );
            }
        }
    }

    Ok(expanded)
}

// =============================================================================
// Prompt-learning config expansion helpers
// =============================================================================

/// Resolve integer seeds from list or range spec.
pub fn resolve_seed_spec(seeds_spec: &Value) -> Result<Vec<i64>, CoreError> {
    match seeds_spec {
        Value::Null => Ok(Vec::new()),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                if let Some(n) = item.as_i64() {
                    out.push(n);
                } else {
                    return Err(CoreError::Validation(
                        "seed array must contain integers".to_string(),
                    ));
                }
            }
            Ok(out)
        }
        Value::Object(map) => {
            let start = map.get("start").and_then(|v| v.as_i64()).ok_or_else(|| {
                CoreError::Validation("range dict must include integer 'start'".to_string())
            })?;
            let end = map.get("end").and_then(|v| v.as_i64()).ok_or_else(|| {
                CoreError::Validation("range dict must include integer 'end'".to_string())
            })?;
            let step = map.get("step").and_then(|v| v.as_i64()).unwrap_or(1);
            if step <= 0 {
                return Err(CoreError::Validation(
                    "range dict 'step' must be positive".to_string(),
                ));
            }
            Ok((start..end).step_by(step as usize).collect())
        }
        _ => Err(CoreError::Validation(
            "invalid seeds spec: expected list or range dict".to_string(),
        )),
    }
}

/// Expand minimal eval config to full config.
pub fn expand_eval_config(minimal: &Value) -> Result<Value, CoreError> {
    let map = minimal
        .as_object()
        .ok_or_else(|| CoreError::Validation("config must be an object".to_string()))?;

    let task_app_url = map
        .get("task_app_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CoreError::Validation("task_app_url is required".to_string()))?;

    let seeds_value = map
        .get("seeds")
        .ok_or_else(|| CoreError::Validation("seeds is required".to_string()))?;

    let defaults = expansion_defaults(map.get("defaults_version").and_then(|v| v.as_str()))?;
    let seeds = resolve_seed_spec(seeds_value)?;
    let seeds_len = seeds.len() as i64;
    let env_name = map
        .get("env_name")
        .and_then(|v| v.as_str())
        .or_else(|| map.get("app_id").and_then(|v| v.as_str()))
        .unwrap_or("default");

    let policy = map
        .get("policy")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));

    let mut out = Map::new();
    out.insert(
        "task_app_url".to_string(),
        Value::String(task_app_url.to_string()),
    );
    out.insert("env_name".to_string(), Value::String(env_name.to_string()));
    if let Some(app_id) = map.get("app_id") {
        out.insert("app_id".to_string(), app_id.clone());
    }
    out.insert(
        "seeds".to_string(),
        Value::Array(seeds.into_iter().map(Value::from).collect()),
    );
    out.insert(
        "max_concurrent".to_string(),
        Value::Number((defaults.eval_max_concurrent.min(seeds_len)).into()),
    );
    out.insert(
        "timeout".to_string(),
        Value::Number(
            serde_json::Number::from_f64(defaults.eval_timeout)
                .ok_or_else(|| CoreError::Validation("invalid eval_timeout".to_string()))?,
        ),
    );
    out.insert("policy".to_string(), policy);
    out.insert(
        "_defaults_version".to_string(),
        Value::String(defaults.version),
    );

    Ok(Value::Object(out))
}

fn build_termination_config(minimal: &Map<String, Value>) -> Option<Value> {
    let has_constraint = ["max_cost_usd", "max_rollouts", "max_seconds", "max_trials"]
        .iter()
        .any(|k| minimal.contains_key(*k));

    if !has_constraint {
        return None;
    }

    let mut map = Map::new();
    map.insert(
        "max_cost_usd".to_string(),
        minimal
            .get("max_cost_usd")
            .cloned()
            .unwrap_or_else(|| Value::Number(serde_json::Number::from_f64(1000.0).unwrap())),
    );
    map.insert(
        "max_trials".to_string(),
        minimal
            .get("max_trials")
            .cloned()
            .unwrap_or_else(|| Value::Number(100000.into())),
    );
    if let Some(v) = minimal.get("max_rollouts") {
        map.insert("max_rollouts".to_string(), v.clone());
    }
    if let Some(v) = minimal.get("max_seconds") {
        map.insert("max_seconds".to_string(), v.clone());
    }
    Some(Value::Object(map))
}

/// Expand minimal GEPA config to full config.
pub fn expand_gepa_config(minimal: &Value) -> Result<Value, CoreError> {
    let map = minimal
        .as_object()
        .ok_or_else(|| CoreError::Validation("config must be an object".to_string()))?;

    let task_app_url = map
        .get("task_app_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CoreError::Validation("task_app_url is required".to_string()))?;

    for key in [
        "proposer_effort",
        "proposer_output_tokens",
        "num_generations",
        "children_per_generation",
    ] {
        if !map.contains_key(key) {
            return Err(CoreError::Validation(format!("{} is required", key)));
        }
    }

    let defaults = expansion_defaults(map.get("defaults_version").and_then(|v| v.as_str()))?;

    let (train_seeds, val_seeds) =
        if let Some(total) = map.get("total_seeds").and_then(|v| v.as_i64()) {
            let split = (total as f64 * defaults.train_ratio) as i64;
            let train: Vec<i64> = (0..split).collect();
            let val: Vec<i64> = (split..total).collect();
            (train, val)
        } else if map.contains_key("train_seeds")
            || map.contains_key("validation_seeds")
            || map.contains_key("val_seeds")
        {
            let train_value = map.get("train_seeds").cloned().unwrap_or(Value::Null);
            let val_value = map
                .get("validation_seeds")
                .cloned()
                .or_else(|| map.get("val_seeds").cloned())
                .unwrap_or(Value::Null);
            (
                resolve_seed_spec(&train_value)?,
                resolve_seed_spec(&val_value)?,
            )
        } else {
            return Err(CoreError::Validation(
                "Either total_seeds or (train_seeds + validation_seeds) is required".to_string(),
            ));
        };

    if train_seeds.is_empty() {
        return Err(CoreError::Validation(
            "train_seeds cannot be empty".to_string(),
        ));
    }
    if val_seeds.is_empty() {
        return Err(CoreError::Validation(
            "validation_seeds cannot be empty".to_string(),
        ));
    }

    let n_train = train_seeds.len() as i64;
    let computed = n_train / defaults.pop_size_divisor.max(1);
    let mut pop_size = map
        .get("population_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(computed);
    if pop_size < defaults.pop_size_min {
        pop_size = defaults.pop_size_min;
    }
    if pop_size > defaults.pop_size_max {
        pop_size = defaults.pop_size_max;
    }

    let mut gepa = Map::new();
    let env_name = map
        .get("env_name")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    gepa.insert("env_name".to_string(), Value::String(env_name.to_string()));
    gepa.insert(
        "proposer_effort".to_string(),
        map.get("proposer_effort").cloned().unwrap_or(Value::Null),
    );
    gepa.insert(
        "proposer_output_tokens".to_string(),
        map.get("proposer_output_tokens")
            .cloned()
            .unwrap_or(Value::Null),
    );

    let mut evaluation = Map::new();
    evaluation.insert(
        "train_seeds".to_string(),
        Value::Array(train_seeds.into_iter().map(Value::from).collect()),
    );
    evaluation.insert(
        "validation_seeds".to_string(),
        Value::Array(val_seeds.into_iter().map(Value::from).collect()),
    );
    gepa.insert("evaluation".to_string(), Value::Object(evaluation));

    let mut rollout = Map::new();
    rollout.insert(
        "budget".to_string(),
        Value::Number(defaults.rollout_budget.into()),
    );
    rollout.insert(
        "max_concurrent".to_string(),
        Value::Number(defaults.rollout_max_concurrent.into()),
    );
    gepa.insert("rollout".to_string(), Value::Object(rollout));

    let mut mutation = Map::new();
    mutation.insert(
        "rate".to_string(),
        serde_json::Number::from_f64(defaults.mutation_rate)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Validation("invalid mutation_rate".to_string()))?,
    );
    gepa.insert("mutation".to_string(), Value::Object(mutation));

    let mut population = Map::new();
    population.insert("initial_size".to_string(), Value::Number(pop_size.into()));
    population.insert(
        "num_generations".to_string(),
        map.get("num_generations").cloned().unwrap_or(Value::Null),
    );
    population.insert(
        "children_per_generation".to_string(),
        map.get("children_per_generation")
            .cloned()
            .unwrap_or(Value::Null),
    );
    population.insert(
        "crossover_rate".to_string(),
        serde_json::Number::from_f64(defaults.crossover_rate)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Validation("invalid crossover_rate".to_string()))?,
    );
    population.insert(
        "selection_pressure".to_string(),
        serde_json::Number::from_f64(defaults.selection_pressure)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Validation("invalid selection_pressure".to_string()))?,
    );
    gepa.insert("population".to_string(), Value::Object(population));

    let mut archive = Map::new();
    let archive_size = pop_size * defaults.archive_multiplier;
    archive.insert("size".to_string(), Value::Number(archive_size.into()));
    archive.insert(
        "pareto_set_size".to_string(),
        Value::Number(archive_size.into()),
    );
    archive.insert(
        "pareto_eps".to_string(),
        serde_json::Number::from_f64(defaults.pareto_eps)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Validation("invalid pareto_eps".to_string()))?,
    );
    archive.insert(
        "feedback_fraction".to_string(),
        serde_json::Number::from_f64(defaults.feedback_fraction)
            .map(Value::Number)
            .ok_or_else(|| CoreError::Validation("invalid feedback_fraction".to_string()))?,
    );
    gepa.insert("archive".to_string(), Value::Object(archive));

    let mut out = Map::new();
    out.insert("algorithm".to_string(), Value::String("gepa".to_string()));
    out.insert(
        "task_app_url".to_string(),
        Value::String(task_app_url.to_string()),
    );
    if let Some(task_app_id) = map.get("task_app_id") {
        out.insert("task_app_id".to_string(), task_app_id.clone());
    }
    for key in [
        "policy",
        "env_config",
        "verifier",
        "proxy_models",
        "initial_prompt",
        "auto_discover_patterns",
        "use_byok",
    ] {
        if let Some(value) = map.get(key) {
            if !value.is_null() {
                out.insert(key.to_string(), value.clone());
            }
        }
    }
    out.insert("gepa".to_string(), Value::Object(gepa));
    if let Some(term) = build_termination_config(map) {
        out.insert("termination_config".to_string(), term);
    }
    out.insert(
        "_defaults_version".to_string(),
        Value::String(defaults.version),
    );

    Ok(Value::Object(out))
}

/// Check whether a config appears to be minimal and needs expansion.
pub fn is_minimal_config(config: &Value) -> bool {
    let map = match config.as_object() {
        Some(map) => map,
        None => return false,
    };

    let has_minimal = map.contains_key("total_seeds") || map.contains_key("defaults_version");
    let has_full = map.contains_key("gepa") || map.contains_key("mipro");

    has_minimal && !has_full
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deep_merge_objects() {
        let mut base = json!({
            "a": 1,
            "b": {
                "c": 2,
                "d": 3
            }
        });
        let overrides = json!({
            "b": {
                "c": 99
            },
            "e": 4
        });

        deep_merge(&mut base, &overrides);

        assert_eq!(base["a"], 1);
        assert_eq!(base["b"]["c"], 99);
        assert_eq!(base["b"]["d"], 3);
        assert_eq!(base["e"], 4);
    }

    #[test]
    fn test_deep_merge_replace() {
        let mut base = json!({ "a": [1, 2, 3] });
        let overrides = json!({ "a": [4, 5] });

        deep_merge(&mut base, &overrides);

        assert_eq!(base["a"], json!([4, 5]));
    }

    #[test]
    fn test_deep_update_dot_keys() {
        let mut base = json!({
            "prompt_learning": {
                "policy": { "model": "a" }
            }
        });
        let overrides = json!({
            "prompt_learning.policy.model": "b",
            "prompt_learning.gepa.rollout.budget": 10
        });

        deep_update(&mut base, &overrides);

        assert_eq!(base["prompt_learning"]["policy"]["model"], "b");
        assert_eq!(base["prompt_learning"]["gepa"]["rollout"]["budget"], 10);
    }

    #[test]
    fn test_deep_update_nested_merge() {
        let mut base = json!({
            "a": { "b": 1, "c": 2 }
        });
        let overrides = json!({
            "a": { "b": 3 }
        });

        deep_update(&mut base, &overrides);

        assert_eq!(base["a"]["b"], 3);
        assert_eq!(base["a"]["c"], 2);
    }

    #[test]
    fn test_resolve_seeds_array() {
        let seeds = json!(["seed1", "seed2", "seed3"]);
        let result = resolve_seeds(&seeds).unwrap();
        assert_eq!(result, vec!["seed1", "seed2", "seed3"]);
    }

    #[test]
    fn test_resolve_seeds_object() {
        let seeds = json!({
            "seed1": {"data": 1},
            "seed2": {"data": 2}
        });
        let mut result = resolve_seeds(&seeds).unwrap();
        result.sort();
        assert_eq!(result, vec!["seed1", "seed2"]);
    }

    #[test]
    fn test_split_train_validation() {
        let seeds: Vec<String> = (1..=10).map(|i| format!("seed{}", i)).collect();

        let (train, val) = split_train_validation(&seeds, 0.7);
        assert_eq!(train.len(), 7);
        assert_eq!(val.len(), 3);

        let (train, val) = split_train_validation(&seeds, 0.5);
        assert_eq!(train.len(), 5);
        assert_eq!(val.len(), 5);
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
            [optimization]
            generations = 10
            mutation_rate = 0.2

            [optimization.nested]
            value = "test"
        "#;

        let result = parse_toml(toml).unwrap();
        assert_eq!(result["optimization"]["generations"], 10);
        assert_eq!(result["optimization"]["mutation_rate"], 0.2);
        assert_eq!(result["optimization"]["nested"]["value"], "test");
    }

    #[test]
    fn test_validate_overrides() {
        let base = json!({
            "a": 1,
            "b": {
                "c": 2
            }
        });

        // Valid override
        let valid = json!({
            "a": 99,
            "b": {
                "c": 100
            }
        });
        assert!(validate_overrides(&base, &valid, "").is_ok());

        // Invalid override (unknown key)
        let invalid = json!({
            "unknown_key": 1
        });
        assert!(validate_overrides(&base, &invalid, "").is_err());
    }
}
