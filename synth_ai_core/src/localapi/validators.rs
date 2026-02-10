use crate::errors::CoreError;
use crate::urls;
use serde_json::Value;
use url::Url;

pub fn validate_rollout_response_for_rl(response: &Value) -> Vec<String> {
    let mut issues = Vec::new();

    let trace_block = response.get("trace");
    let mut event_history: Option<&Value> = None;
    if let Some(Value::Object(trace)) = trace_block {
        if let Some(ev) = trace.get("event_history") {
            event_history = Some(ev);
        } else if let Some(Value::Object(session_trace)) = trace.get("session_trace") {
            event_history = session_trace.get("event_history");
        }
    }

    let has_event_history = matches!(event_history, Some(Value::Array(list)) if !list.is_empty());

    let mut trace_correlation_id = response.get("trace_correlation_id");
    if trace_correlation_id.is_none() {
        if let Some(Value::Object(trace)) = trace_block {
            if let Some(Value::Object(meta)) = trace.get("metadata") {
                trace_correlation_id = meta.get("trace_correlation_id");
            }
        }
    }

    if !matches!(trace_correlation_id, Some(Value::String(s)) if !s.trim().is_empty()) {
        issues.push(
            "Missing trace_correlation_id (top-level or trace.metadata). RL trainer requires this to link traces."
                .to_string(),
        );
    }

    if !has_event_history {
        issues.push(
            "trace.event_history is missing or empty. Return a v3/v4 trace or provide inference_url for hydration."
                .to_string(),
        );
    }

    if !has_event_history {
        match response.get("inference_url") {
            None => {
                issues.push(
                    "inference_url is missing. RL trainer needs this to hydrate traces when event_history is absent."
                        .to_string(),
                );
            }
            Some(Value::String(url)) => {
                if !url.contains("?cid=") {
                    issues.push(format!(
                        "inference_url should contain '?cid=' for trace correlation. Got: {}...",
                        url.chars().take(80).collect::<String>()
                    ));
                }
            }
            Some(other) => {
                let type_name = match other {
                    Value::Null => "null",
                    Value::Bool(_) => "bool",
                    Value::Number(_) => "number",
                    Value::String(_) => "string",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                };
                issues.push(format!(
                    "inference_url must be a string, got: {}",
                    type_name
                ));
            }
        }
    }

    issues
}

pub fn normalize_inference_url(url: Option<&str>, default: &str) -> Result<String, CoreError> {
    let mut candidate = url.unwrap_or(default).trim().to_string();
    if candidate.is_empty() {
        candidate = default.to_string();
    }

    let mut parsed = Url::parse(&candidate)?;
    let mut path = parsed.path().trim_end_matches('/').to_string();
    let mut query = parsed.query().unwrap_or("").to_string();

    if !query.is_empty() && query.contains('/') {
        let mut parts = query.splitn(2, '/');
        let base_query = parts.next().unwrap_or("");
        let remainder = parts.next().unwrap_or("");
        let mut remainder_path = remainder.to_string();
        let mut extra_query = String::new();
        for separator in ["&", "?"] {
            if let Some(idx) = remainder_path.find(separator) {
                extra_query = remainder_path[idx + 1..].to_string();
                remainder_path = remainder_path[..idx].to_string();
                break;
            }
        }
        let query_path = format!("/{}", remainder_path.trim_start_matches('/'));
        let mut merged_query = Vec::new();
        if !base_query.is_empty() {
            merged_query.push(base_query.to_string());
        }
        if !extra_query.is_empty() {
            merged_query.push(extra_query);
        }
        let merged_query = merged_query.join("&");

        if !query_path.is_empty() && query_path != "/" {
            path = format!("{}{}", path.trim_end_matches('/'), query_path);
        }
        parsed.set_path(path.as_str());
        parsed.set_query(if merged_query.is_empty() {
            None
        } else {
            Some(&merged_query)
        });
        query = parsed.query().unwrap_or("").to_string();
        path = parsed.path().trim_end_matches('/').to_string();
    }

    if path.ends_with("/v1/chat/completions") || path.ends_with("/chat/completions") {
        if !query.is_empty() && query.contains('/') {
            let trimmed = query.splitn(2, '/').next().unwrap_or("");
            parsed.set_query(if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            });
        }
        return Ok(parsed.to_string());
    }

    let new_path = if (path.contains("/v1/") && !path.ends_with("/v1")) || path.ends_with("/v1") {
        format!("{}/chat/completions", path)
    } else if path.ends_with("/chat") {
        format!("{}/completions", path)
    } else if path.is_empty() {
        "/v1/chat/completions".to_string()
    } else {
        format!("{}/v1/chat/completions", path)
    };

    parsed.set_path(new_path.as_str());
    if !query.is_empty() && query.contains('/') {
        let trimmed = query.splitn(2, '/').next().unwrap_or("");
        parsed.set_query(if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        });
    }

    Ok(parsed.to_string())
}

pub fn validate_task_app_url(url: &str) -> Result<String, CoreError> {
    let parsed = urls::validate_task_app_url(url)?;
    Ok(parsed.to_string().trim_end_matches('/').to_string())
}
