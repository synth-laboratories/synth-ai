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

    let evaluation = gepa
        .get_mut("evaluation")
        .ok_or_else(|| {
            CoreError::Validation(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
                    .to_string(),
            )
        })?
        .as_object_mut()
        .ok_or_else(|| {
            CoreError::Validation(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
                    .to_string(),
            )
        })?;

    let train_seeds = evaluation
        .get("train_seeds")
        .or_else(|| evaluation.get("seeds"))
        .cloned();

    if value_is_missing(train_seeds.as_ref()) {
        return Err(CoreError::Validation(
            "GEPA config missing train_seeds: [prompt_learning.gepa.evaluation] must have 'train_seeds' or 'seeds' field"
                .to_string(),
        ));
    }

    let val_seeds = evaluation
        .get("val_seeds")
        .or_else(|| evaluation.get("validation_seeds"))
        .cloned();

    if value_is_missing(val_seeds.as_ref()) {
        return Err(CoreError::Validation(
            "GEPA config missing val_seeds: [prompt_learning.gepa.evaluation] must have 'val_seeds' or 'validation_seeds' field"
                .to_string(),
        ));
    }

    set_if_missing(pl_map, "train_seeds", train_seeds.clone());
    set_if_missing(pl_map, "evaluation_seeds", train_seeds);

    Ok(())
}

fn normalize_mipro(pl_map: &mut Map<String, Value>) -> Result<(), CoreError> {
    let fallback_bootstrap = pl_map.get("bootstrap_train_seeds").cloned();
    let fallback_online = pl_map.get("online_pool").cloned();
    let fallback_test = pl_map.get("test_pool").cloned();
    let fallback_reference = pl_map.get("reference_pool").cloned();
    let needs_env_name = pl_map.get("env_name").is_none() && pl_map.get("task_app_id").is_none();

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
            bootstrap.or_else(|| fallback_bootstrap.clone()),
        );
        set_if_missing(
            mipro_map,
            "online_pool",
            online_pool.or_else(|| fallback_online.clone()),
        );
        set_if_missing(
            mipro_map,
            "test_pool",
            test_pool.or_else(|| fallback_test.clone()),
        );
        set_if_missing(
            mipro_map,
            "reference_pool",
            reference_pool.or_else(|| fallback_reference.clone()),
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
        .get("task_app_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let env_task_url = std::env::var("TASK_APP_URL").ok();

    let final_task_url = if let Some(cli) = cli_task_url {
        cli
    } else if let Some(cfg) = config_task_url {
        cfg
    } else if let Some(env) = env_task_url {
        env
    } else {
        return Err(CoreError::Validation(
            "task_app_url is required".to_string(),
        ));
    };

    pl_map.insert(
        "task_app_url".to_string(),
        Value::String(final_task_url.clone()),
    );

    let cli_api_key = overrides_obj
        .get("task_app_api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let config_api_key = pl_map
        .get("task_app_api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let env_api_key = std::env::var("ENVIRONMENT_API_KEY").ok();
    let api_key = cli_api_key.or(env_api_key).or(config_api_key);
    if api_key.is_none() {
        return Err(CoreError::Validation(
            "task_app_api_key is required".to_string(),
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

    Ok((Value::Object(payload), final_task_url))
}
