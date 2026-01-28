//! Model identifier utilities shared across SDKs.

use crate::errors::CoreError;
use serde_json::{Map, Value};

/// Normalize a model identifier for API requests.
pub fn normalize_model_identifier(
    model: &str,
    allow_finetuned_prefixes: bool,
) -> Result<String, CoreError> {
    let _ = allow_finetuned_prefixes;
    let normalized = model.trim();
    if normalized.is_empty() {
        return Err(CoreError::Validation(
            "Model identifier cannot be empty".to_string(),
        ));
    }

    let normalized = if normalized.contains('/') {
        normalized.to_string()
    } else {
        normalized.to_lowercase()
    };

    Ok(normalized)
}

/// Detect provider from a model identifier.
pub fn detect_model_provider(model: &str) -> Result<String, CoreError> {
    let model_lower = model.trim().to_lowercase();
    if model_lower.is_empty() {
        return Err(CoreError::Validation(
            "Model identifier cannot be empty".to_string(),
        ));
    }

    let provider = if model_lower.starts_with("gpt-") || model_lower.starts_with("o1") {
        "openai"
    } else if model_lower.starts_with("gemini") {
        "google"
    } else if model_lower.starts_with("claude") {
        "anthropic"
    } else if model_lower.contains("llama") || model_lower.contains("mixtral") {
        "groq"
    } else {
        "openai"
    };

    Ok(provider.to_string())
}

/// Load the supported models configuration from embedded assets.
pub fn supported_models() -> Value {
    let raw = include_str!("../assets/supported_models.json");
    serde_json::from_str(raw).unwrap_or_else(|_| Value::Object(Map::new()))
}
