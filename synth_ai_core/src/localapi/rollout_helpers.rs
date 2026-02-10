use crate::data::{Artifact, SuccessStatus};
use crate::errors::CoreError;
use crate::localapi::types::{RolloutMetrics, RolloutRequest, RolloutResponse};
use serde_json::{Map, Number, Value};

pub fn build_rollout_response(
    request: &RolloutRequest,
    outcome_reward: f64,
    inference_url: Option<&str>,
    trace: Option<Value>,
    policy_config: Option<&Value>,
    artifact: Option<Vec<Artifact>>,
    success_status: Option<SuccessStatus>,
    status_detail: Option<String>,
    reward_details: Option<&Value>,
) -> Result<RolloutResponse, CoreError> {
    let mut reward_map = Map::new();
    reward_map.insert(
        "outcome_reward".to_string(),
        Value::Number(Number::from_f64(outcome_reward).unwrap_or(Number::from(0))),
    );
    if let Some(Value::Object(extra)) = reward_details {
        for (k, v) in extra.iter() {
            reward_map.insert(k.clone(), v.clone());
        }
    }

    // Auto-derive outcome_objectives if not provided in reward_details
    if !reward_map.contains_key("outcome_objectives") {
        let mut objectives_map = Map::new();
        objectives_map.insert(
            "reward".to_string(),
            Value::Number(Number::from_f64(outcome_reward).unwrap_or(Number::from(0))),
        );
        reward_map.insert(
            "outcome_objectives".to_string(),
            Value::Object(objectives_map),
        );
    }

    let reward_info: RolloutMetrics = serde_json::from_value(Value::Object(reward_map))
        .map_err(|e| CoreError::InvalidInput(e.to_string()))?;

    let resolved_inference = if let Some(url) = inference_url {
        Some(url.to_string())
    } else if let Some(Value::Object(cfg)) = policy_config {
        cfg.get("inference_url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    Ok(RolloutResponse {
        trace_correlation_id: request.trace_correlation_id.clone(),
        reward_info,
        trace,
        inference_url: resolved_inference,
        artifact,
        success_status,
        status_detail,
        override_application_results: None,
    })
}
