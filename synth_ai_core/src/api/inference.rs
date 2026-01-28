//! Inference API client.

use serde_json::Value;

use crate::errors::CoreError;
use crate::http::HttpClient;
use crate::models::normalize_model_identifier;

/// Inference client for chat completions.
pub struct InferenceClient<'a> {
    http: &'a HttpClient,
}

impl<'a> InferenceClient<'a> {
    pub(crate) fn new(http: &'a HttpClient) -> Self {
        Self { http }
    }

    /// Create a chat completion request through Synth inference proxy.
    pub async fn chat_completion(&self, mut body: Value) -> Result<Value, CoreError> {
        let obj = body
            .as_object_mut()
            .ok_or_else(|| CoreError::Validation("request must be an object".to_string()))?;

        let model = obj
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::Validation("model is required".to_string()))?;
        let normalized = normalize_model_identifier(model, false)?;
        obj.insert("model".to_string(), Value::String(normalized));

        if !obj.contains_key("thinking_budget") {
            obj.insert("thinking_budget".to_string(), Value::Number(256.into()));
        }

        self.http
            .post_json("/api/inference/v1/chat/completions", &body)
            .await
            .map_err(CoreError::from)
    }
}
