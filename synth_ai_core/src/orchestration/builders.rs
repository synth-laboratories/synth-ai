//! Prompt learning payload builders.
//!
//! These helpers consolidate config normalization and override merging so
//! both Rust and Python SDKs share the same core logic.

use serde_json::{Map, Value};

use crate::config::deep_update;
use crate::errors::CoreError;

fn ensure_api_base(base_url: &str) -> String {
    let mut normalized = base_url.trim().trim_end_matches('/').to_string();
    if normalized.ends_with("/api") {
        let trimmed = normalized.trim_end_matches("/api");
        normalized = trimmed.trim_end_matches('/').to_string();
    }
    if normalized.ends_with("/v1") {
        let trimmed = normalized.trim_end_matches("/v1");
        normalized = trimmed.trim_end_matches('/').to_string();
    }
    format!("{}/api", normalized)
}

fn prompt_learning_section_mut(config: &mut Value) -> Result<&mut Map<String, Value>, CoreError> {
    let map = config
        .as_object_mut()
        .ok_or_else(|| CoreError::Validation("config must be an object".to_string()))?;

    if !map.contains_key("prompt_learning") {
        let existing = std::mem::take(map);
        let mut pl_map = Map::new();
        for (k, v) in existing {
            pl_map.insert(k, v);
        }
        map.insert("prompt_learning".to_string(), Value::Object(pl_map));
    }

    match map.get_mut("prompt_learning") {
        Some(Value::Object(pl_map)) => Ok(pl_map),
        _ => Err(CoreError::Validation(
            "prompt_learning must be an object".to_string(),
        )),
    }
}

fn value_is_missing(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(Value::Null) => true,
        _ => false,
    }
}

fn set_if_missing(map: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if value.is_none() {
        return;
    }
    let missing = value_is_missing(map.get(key));
    if missing {
        map.insert(key.to_string(), value.unwrap());
    }
}

fn normalize_gepa(pl_map: &mut Map<String, Value>) -> Result<(), CoreError> {
    let gepa = pl_map
        .get_mut("gepa")
        .ok_or_else(|| {
            CoreError::Validation(
                "GEPA config missing: [prompt_learning.gepa] section is required".to_string(),
            )
        })?
        .as_object_mut()
        .ok_or_else(|| {
            CoreError::Validation(
                "GEPA config missing: [prompt_learning.gepa] section is required".to_string(),
            )
        })?;

    let mut train_seeds: Option<Value> = None;
    let mut val_seeds: Option<Value> = None;

    if let Some(evaluation_obj) = gepa
        .get_mut("evaluation")
        .and_then(|value| value.as_object_mut())
    {
        train_seeds = evaluation_obj
            .get("train_seeds")
            .or_else(|| evaluation_obj.get("seeds"))
            .cloned();
        val_seeds = evaluation_obj
            .get("val_seeds")
            .or_else(|| evaluation_obj.get("validation_seeds"))
            .cloned();
    }

    if value_is_missing(train_seeds.as_ref()) || value_is_missing(val_seeds.as_ref()) {
        if let Some(task_data) = pl_map.get("task_data").and_then(|value| value.as_object()) {
            if value_is_missing(train_seeds.as_ref()) {
                let reflection = task_data
                    .get("train_pools")
                    .and_then(|v| v.as_object())
                    .and_then(|p| p.get("reflection_seeds"))
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let pareto = task_data
                    .get("train_pools")
                    .and_then(|v| v.as_object())
                    .and_then(|p| p.get("pareto_seeds"))
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let mut merged = reflection;
                for seed in pareto {
                    if !merged.iter().any(|existing| existing == &seed) {
                        merged.push(seed);
                    }
                }
                if !merged.is_empty() {
                    train_seeds = Some(Value::Array(merged));
                }
            }
            if value_is_missing(val_seeds.as_ref()) {
                val_seeds = task_data
                    .get("validation_pools")
                    .and_then(|v| v.as_object())
                    .and_then(|p| p.get("main_seeds"))
                    .cloned()
                    .or_else(|| task_data.get("validation_seeds").cloned());
            }
        }
    }

    if value_is_missing(train_seeds.as_ref()) {
        return Err(CoreError::Validation(
            "GEPA config missing train_seeds: provide [prompt_learning.task_data.train_pools] (reflection_seeds/pareto_seeds) or [prompt_learning.gepa.evaluation] seeds"
                .to_string(),
        ));
    }

    if value_is_missing(val_seeds.as_ref()) {
        return Err(CoreError::Validation(
            "GEPA config missing val_seeds: provide prompt_learning.task_data.validation_seeds or [prompt_learning.gepa.evaluation].validation_seeds"
                .to_string(),
        ));
    }

    set_if_missing(pl_map, "train_seeds", train_seeds.clone());
    set_if_missing(pl_map, "evaluation_seeds", train_seeds);

    Ok(())
}

fn normalize_mipro(pl_map: &mut Map<String, Value>) -> Result<(), CoreError> {
    let strict_bootstrap = pl_map.get("bootstrap_train_seeds").cloned();
    let strict_online = pl_map.get("online_pool").cloned();
    let strict_test = pl_map.get("test_pool").cloned();
    let strict_reference = pl_map.get("reference_pool").cloned();
    let needs_env_name = pl_map.get("env_name").is_none() && pl_map.get("container_id").is_none();

    let env_name = {
        let mipro = pl_map.get_mut("mipro").ok_or_else(|| {
            CoreError::Validation(
                "MIPRO config missing [prompt_learning.mipro] section.".to_string(),
            )
        })?;
        if !mipro.is_object() {
            *mipro = Value::Object(Map::new());
        }
        let mipro_map = mipro.as_object_mut().expect("mipro is object");

        let bootstrap = mipro_map.get("bootstrap_train_seeds").cloned();
        let online_pool = mipro_map.get("online_pool").cloned();
        let test_pool = mipro_map.get("test_pool").cloned();
        let reference_pool = mipro_map.get("reference_pool").cloned();

        set_if_missing(
            mipro_map,
            "bootstrap_train_seeds",
            bootstrap.or_else(|| strict_bootstrap.clone()),
        );
        set_if_missing(
            mipro_map,
            "online_pool",
            online_pool.or_else(|| strict_online.clone()),
        );
        set_if_missing(
            mipro_map,
            "test_pool",
            test_pool.or_else(|| strict_test.clone()),
        );
        set_if_missing(
            mipro_map,
            "reference_pool",
            reference_pool.or_else(|| strict_reference.clone()),
        );

        if value_is_missing(mipro_map.get("bootstrap_train_seeds")) {
            return Err(CoreError::Validation(
                "MIPRO config missing bootstrap_train_seeds.".to_string(),
            ));
        }
        if value_is_missing(mipro_map.get("online_pool")) {
            return Err(CoreError::Validation(
                "MIPRO config missing online_pool.".to_string(),
            ));
        }

        if needs_env_name {
            mipro_map
                .get("env_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    };

    if let Some(env_name) = env_name {
        pl_map.insert("env_name".to_string(), Value::String(env_name));
    }

    Ok(())
}

/// Build a prompt learning payload from a config dict and overrides.
pub fn build_prompt_learning_payload(
    config: &Value,
    task_url: Option<String>,
    overrides: Option<&Value>,
) -> Result<(Value, String), CoreError> {
    let mut config_dict = config.clone();
    let pl_map = prompt_learning_section_mut(&mut config_dict)?;

    let overrides_value = overrides
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let overrides_obj = overrides_value
        .as_object()
        .cloned()
        .unwrap_or_else(Map::new);

    let cli_task_url = overrides_obj
        .get("task_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or(task_url);
    let config_task_url = pl_map
        .get("container_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let has_container_id = pl_map
        .get("container_id")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let env_task_url = std::env::var("CONTAINER_URL").ok();

    let final_task_url = if let Some(cli) = cli_task_url {
        Some(cli)
    } else if let Some(cfg) = config_task_url {
        Some(cfg)
    } else if let Some(env) = env_task_url {
        Some(env)
    } else {
        None
    };

    if let Some(url) = final_task_url.clone() {
        pl_map.insert("container_url".to_string(), Value::String(url));
    } else if !has_container_id {
        return Err(CoreError::Validation(
            "container_url or container_id is required".to_string(),
        ));
    }

    let algorithm = pl_map
        .get("algorithm")
        .and_then(|v| v.as_str())
        .unwrap_or("gepa")
        .to_string();

    if algorithm == "gepa" {
        normalize_gepa(pl_map)?;
    } else if algorithm == "mipro" {
        normalize_mipro(pl_map)?;
    }

    let config_overrides_src = if let Some(Value::Object(inner)) = overrides_obj.get("overrides") {
        inner.clone()
    } else {
        overrides_obj.clone()
    };

    let mut config_overrides = Map::new();
    for (k, v) in config_overrides_src {
        if k == "backend" || k == "task_url" || k == "metadata" || k == "auto_start" {
            continue;
        }
        config_overrides.insert(k, v);
    }

    if !config_overrides.is_empty() {
        deep_update(&mut config_dict, &Value::Object(config_overrides.clone()));
    }

    if algorithm == "mipro" {
        let pl_map = prompt_learning_section_mut(&mut config_dict)?;
        normalize_mipro(pl_map)?;
    }

    if !config_dict
        .as_object()
        .map(|m| m.contains_key("prompt_learning"))
        .unwrap_or(false)
    {
        return Err(CoreError::Validation(
            "config_dict must have 'prompt_learning' key".to_string(),
        ));
    }

    let mut metadata = overrides_obj
        .get("metadata")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_else(Map::new);

    if let Some(Value::String(backend)) = overrides_obj.get("backend") {
        let api_base = ensure_api_base(backend);
        metadata.insert("backend_base_url".to_string(), Value::String(api_base));
    }

    let auto_start = overrides_obj
        .get("auto_start")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut payload = Map::new();
    payload.insert("algorithm".to_string(), Value::String(algorithm));
    payload.insert("config_body".to_string(), config_dict);
    payload.insert("overrides".to_string(), Value::Object(config_overrides));
    payload.insert("metadata".to_string(), Value::Object(metadata));
    payload.insert("auto_start".to_string(), Value::Bool(auto_start));

    Ok((Value::Object(payload), final_task_url.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::build_prompt_learning_payload;
    use serde_json::json;

    #[test]
    fn builds_prompt_learning_payload() {
        let config = json!({
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": "http://localhost:8782",
                "policy": {"provider": "openai", "model": "gpt-4o-mini"},
                "gepa": {
                    "evaluation": {"train_seeds": [0, 1], "val_seeds": [2, 3]}
                }
            }
        });
        let (payload, task_url) = build_prompt_learning_payload(
            &config,
            None,
            Some(&json!({"metadata": {"source": "test"}, "auto_start": false})),
        )
        .expect("build");
        assert_eq!(task_url, "http://localhost:8782");
        assert_eq!(payload.get("algorithm"), Some(&json!("gepa")));
        assert_eq!(payload.get("auto_start"), Some(&json!(false)));
        let metadata = payload
            .get("metadata")
            .and_then(|v| v.as_object())
            .expect("metadata object");
        assert_eq!(metadata.get("source"), Some(&json!("test")));
    }

    #[test]
    fn applies_backend_override_metadata() {
        let config = json!({
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": "http://localhost:8782",
                "policy": {"provider": "openai", "model": "gpt-4o-mini"},
                "gepa": {
                    "evaluation": {"train_seeds": [0, 1], "val_seeds": [2, 3]}
                }
            }
        });
        let (payload, _task_url) = build_prompt_learning_payload(
            &config,
            None,
            Some(&json!({"backend": "http://127.0.0.1:8080/api"})),
        )
        .expect("payload build");
        let metadata = payload
            .get("metadata")
            .and_then(|v| v.as_object())
            .expect("metadata object");
        assert_eq!(
            metadata.get("backend_base_url"),
            Some(&json!("http://127.0.0.1:8080/api"))
        );
    }
}
