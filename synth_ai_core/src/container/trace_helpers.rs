use crate::errors::CoreError;
use chrono::Utc;
use serde_json::{Map, Value};
use url::Url;
use uuid::Uuid;

fn parse_url(inference_url: &str) -> Option<Url> {
    if let Ok(parsed) = Url::parse(inference_url) {
        return Some(parsed);
    }
    if inference_url.starts_with('/') {
        let candidate = format!("http://localhost{}", inference_url);
        return Url::parse(&candidate).ok();
    }
    if !inference_url.contains("://") {
        let candidate = format!("http://{}", inference_url);
        return Url::parse(&candidate).ok();
    }
    None
}

pub fn extract_trace_correlation_id(inference_url: Option<&str>) -> Option<String> {
    let raw = inference_url?.trim();
    if raw.is_empty() {
        return None;
    }

    let parsed = parse_url(raw)?;
    let path_segments: Vec<&str> = parsed.path().split('/').filter(|s| !s.is_empty()).collect();

    if path_segments.len() >= 3 {
        let tail = &path_segments[path_segments.len() - 2..];
        if tail == ["chat", "completions"] {
            if let Some(segment) = path_segments.get(path_segments.len() - 3) {
                if segment.starts_with("trace_")
                    || segment.starts_with("cid_")
                    || segment.starts_with("eval_")
                {
                    return Some(segment.to_string());
                }
            }
        }
    }

    for segment in path_segments.iter().rev() {
        if segment.starts_with("trace_")
            || segment.starts_with("cid_")
            || segment.starts_with("eval_")
        {
            return Some(segment.to_string());
        }
    }

    let query = parsed.query().unwrap_or("");
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let key = key.to_string();
        if key == "cid" || key == "trace_correlation_id" || key == "trace" {
            let val = value.to_string();
            if !val.trim().is_empty() {
                return Some(val);
            }
        }
    }

    None
}

pub fn validate_trace_correlation_id(
    trace_correlation_id: Option<&str>,
    inference_url: Option<&str>,
    fatal: bool,
) -> Result<Option<String>, CoreError> {
    if let Some(cid) = trace_correlation_id {
        if !cid.trim().is_empty() {
            return Ok(Some(cid.to_string()));
        }
    }

    let inference_url = inference_url.unwrap_or("NOT_SET");
    let error_msg = format!(
        "CRITICAL: Cannot extract trace_correlation_id!\n\nID: UNKNOWN\nInference URL: {}\n\nChecked:\n1. inference_url path segments\n2. inference_url query params\n\nTask app CANNOT proceed without trace_correlation_id.\nThis indicates the RL trainer is not sending it correctly.\n\nSee monorepo/trace_creation_and_judgement.txt 'Fatal Guards' section.\n",
        inference_url
    );

    if fatal {
        return Err(CoreError::InvalidInput(error_msg));
    }

    Ok(None)
}

pub fn include_trace_correlation_id_in_response(
    response_data: &Value,
    trace_correlation_id: Option<&str>,
) -> Value {
    let cid = match trace_correlation_id {
        Some(value) if !value.trim().is_empty() => value.trim(),
        _ => return response_data.clone(),
    };

    let mut output = response_data.clone();
    let obj = match output.as_object_mut() {
        Some(obj) => obj,
        None => return output,
    };

    obj.entry("trace_correlation_id".to_string())
        .or_insert_with(|| Value::String(cid.to_string()));

    if let Some(trace_val) = obj.get_mut("trace") {
        if let Some(trace_obj) = trace_val.as_object_mut() {
            let meta = trace_obj
                .entry("metadata".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(meta_obj) = meta.as_object_mut() {
                meta_obj
                    .entry("trace_correlation_id".to_string())
                    .or_insert_with(|| Value::String(cid.to_string()));
                let corr_ids = meta_obj
                    .entry("correlation_ids".to_string())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(corr_map) = corr_ids.as_object_mut() {
                    corr_map
                        .entry("trace_correlation_id".to_string())
                        .or_insert_with(|| Value::String(cid.to_string()));
                }
            }

            if let Some(session_trace) = trace_obj.get_mut("session_trace") {
                if let Some(session_obj) = session_trace.as_object_mut() {
                    let session_meta = session_obj
                        .entry("metadata".to_string())
                        .or_insert_with(|| Value::Object(Map::new()));
                    if let Some(session_meta_obj) = session_meta.as_object_mut() {
                        session_meta_obj
                            .entry("trace_correlation_id".to_string())
                            .or_insert_with(|| Value::String(cid.to_string()));
                    }
                }
            }
        }
    }

    output
}

pub fn build_trace_payload(
    messages: &Value,
    response: Option<&Value>,
    correlation_id: Option<&str>,
    session_id: Option<&str>,
    metadata: Option<&Value>,
) -> Value {
    let messages_value = match messages {
        Value::Array(list) => Value::Array(list.clone()),
        _ => Value::Array(Vec::new()),
    };

    let llm_response = match response {
        Some(Value::Object(obj)) => {
            if obj.contains_key("message") {
                Value::Object(obj.clone())
            } else if let Some(Value::Array(choices)) = obj.get("choices") {
                if let Some(Value::Object(first)) = choices.get(0) {
                    let mut map = Map::new();
                    map.insert(
                        "message".to_string(),
                        first.get("message").cloned().unwrap_or(Value::Null),
                    );
                    map.insert(
                        "usage".to_string(),
                        obj.get("usage")
                            .cloned()
                            .unwrap_or(Value::Object(Map::new())),
                    );
                    map.insert(
                        "finish_reason".to_string(),
                        first.get("finish_reason").cloned().unwrap_or(Value::Null),
                    );
                    Value::Object(map)
                } else {
                    Value::Object(obj.clone())
                }
            } else {
                Value::Object(obj.clone())
            }
        }
        Some(other) => other.clone(),
        None => Value::Object(Map::new()),
    };

    let mut llm_event = Map::new();
    llm_event.insert("type".to_string(), Value::String("lm_call".to_string()));
    llm_event.insert(
        "event_type".to_string(),
        Value::String("lm_call".to_string()),
    );
    llm_event.insert(
        "timestamp".to_string(),
        Value::String(Utc::now().to_rfc3339()),
    );
    let mut llm_request = Map::new();
    llm_request.insert("messages".to_string(), messages_value);
    llm_event.insert("llm_request".to_string(), Value::Object(llm_request));
    llm_event.insert("llm_response".to_string(), llm_response);
    llm_event.insert("api_format".to_string(), Value::String("chat".to_string()));
    if let Some(cid) = correlation_id {
        if !cid.trim().is_empty() {
            llm_event.insert("correlation_id".to_string(), Value::String(cid.to_string()));
        }
    }

    let mut event_history = Vec::new();
    event_history.push(Value::Object(llm_event));

    let mut trace_meta = match metadata {
        Some(Value::Object(obj)) => obj.clone(),
        _ => Map::new(),
    };
    let session_id = session_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    trace_meta
        .entry("session_id".to_string())
        .or_insert(Value::String(session_id));

    if let Some(cid) = correlation_id {
        if !cid.trim().is_empty() {
            trace_meta
                .entry("trace_correlation_id".to_string())
                .or_insert(Value::String(cid.to_string()));
            let corr_ids = trace_meta
                .entry("correlation_ids".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(corr_map) = corr_ids.as_object_mut() {
                corr_map
                    .entry("trace_correlation_id".to_string())
                    .or_insert(Value::String(cid.to_string()));
            }
        }
    }

    let mut trace = Map::new();
    trace.insert(
        "schema_version".to_string(),
        Value::String("4.0".to_string()),
    );
    trace.insert("event_history".to_string(), Value::Array(event_history));
    trace.insert(
        "markov_blanket_message_history".to_string(),
        Value::Array(Vec::new()),
    );
    trace.insert("metadata".to_string(), Value::Object(trace_meta));

    Value::Object(trace)
}

pub fn build_trajectory_trace(
    messages: &Value,
    response: Option<&Value>,
    correlation_id: Option<&str>,
    session_id: Option<&str>,
    metadata: Option<&Value>,
) -> Value {
    build_trace_payload(messages, response, correlation_id, session_id, metadata)
}

pub fn include_event_history_in_response(
    response_data: &Value,
    messages: Option<&Value>,
    response: Option<&Value>,
    run_id: &str,
    correlation_id: Option<&str>,
) -> Value {
    let mut output = response_data.clone();
    let obj = match output.as_object_mut() {
        Some(obj) => obj,
        None => return output,
    };

    let trace_val = obj
        .entry("trace".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let trace_obj = match trace_val.as_object_mut() {
        Some(obj) => obj,
        None => return output,
    };

    let mut event_history = trace_obj.get("event_history");
    if event_history.is_none() {
        if let Some(Value::Object(session_trace)) = trace_obj.get("session_trace") {
            event_history = session_trace.get("event_history");
        }
    }
    if matches!(event_history, Some(Value::Array(list)) if !list.is_empty()) {
        return output;
    }

    let meta_map = {
        let mut map = Map::new();
        map.insert("run_id".to_string(), Value::String(run_id.to_string()));
        Value::Object(map)
    };

    let new_trace = build_trace_payload(
        messages.unwrap_or(&Value::Array(Vec::new())),
        response,
        correlation_id,
        None,
        Some(&meta_map),
    );

    if let Some(new_trace_obj) = new_trace.as_object() {
        let incoming_meta = trace_obj.get("metadata");
        let mut merged_meta = match new_trace_obj.get("metadata") {
            Some(Value::Object(meta)) => meta.clone(),
            _ => Map::new(),
        };
        if let Some(Value::Object(existing)) = incoming_meta {
            for (k, v) in existing.iter() {
                merged_meta.insert(k.clone(), v.clone());
            }
        }
        trace_obj.insert("metadata".to_string(), Value::Object(merged_meta));
        if let Some(schema) = new_trace_obj.get("schema_version") {
            trace_obj
                .entry("schema_version".to_string())
                .or_insert_with(|| schema.clone());
        }
        if let Some(events) = new_trace_obj.get("event_history") {
            trace_obj.insert("event_history".to_string(), events.clone());
        }
        if let Some(blanket) = new_trace_obj.get("markov_blanket_message_history") {
            trace_obj
                .entry("markov_blanket_message_history".to_string())
                .or_insert_with(|| blanket.clone());
        }

        let events_clone = trace_obj.get("event_history").cloned();
        if let Some(Value::Object(session_trace)) = trace_obj.get_mut("session_trace") {
            if !session_trace.contains_key("event_history") {
                if let Some(events) = events_clone {
                    session_trace.insert("event_history".to_string(), events);
                }
            }
        }
    }

    output
}

pub fn include_event_history_in_trajectories(
    response_data: &Value,
    messages_by_trajectory: Option<&Value>,
    responses_by_trajectory: Option<&Value>,
    run_id: &str,
    correlation_id: Option<&str>,
) -> Value {
    let messages = messages_by_trajectory
        .and_then(|value| value.as_array())
        .and_then(|arr| arr.get(0));
    let response = responses_by_trajectory
        .and_then(|value| value.as_array())
        .and_then(|arr| arr.get(0));

    include_event_history_in_response(response_data, messages, response, run_id, correlation_id)
}

pub fn verify_trace_correlation_id_in_response(
    response_data: &Value,
    expected_correlation_id: Option<&str>,
) -> bool {
    let expected = match expected_correlation_id {
        Some(value) if !value.trim().is_empty() => value,
        _ => return false,
    };

    let obj = match response_data.as_object() {
        Some(obj) => obj,
        None => return false,
    };

    if obj.get("trace_correlation_id").and_then(|v| v.as_str()) != Some(expected) {
        return false;
    }

    let trace_val = match obj.get("trace") {
        Some(Value::Object(trace)) => trace,
        _ => return false,
    };

    let mut trace_meta_id = trace_val
        .get("metadata")
        .and_then(|v| v.as_object())
        .and_then(|meta| meta.get("trace_correlation_id"))
        .and_then(|v| v.as_str());

    if trace_meta_id != Some(expected) {
        if let Some(Value::Object(session_trace)) = trace_val.get("session_trace") {
            trace_meta_id = session_trace
                .get("metadata")
                .and_then(|v| v.as_object())
                .and_then(|meta| meta.get("trace_correlation_id"))
                .and_then(|v| v.as_str());
        }
    }

    trace_meta_id == Some(expected)
}
