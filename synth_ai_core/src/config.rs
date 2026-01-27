//! Configuration utilities for Synth SDK.
//!
//! This module provides:
//! - Core SDK configuration (CoreConfig)
//! - TOML file parsing
//! - Config deep merge
//! - Optimization defaults

use crate::errors::CoreError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
        toml::Value::Float(f) => {
            serde_json::Number::from_f64(f)
                .map(Value::Number)
                .ok_or_else(|| CoreError::Config("invalid float value".to_string()))
        }
        toml::Value::Boolean(b) => Ok(Value::Bool(b)),
        toml::Value::Datetime(dt) => Ok(Value::String(dt.to_string())),
        toml::Value::Array(arr) => {
            let json_arr: Result<Vec<Value>, CoreError> = arr.into_iter().map(toml_to_json).collect();
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

