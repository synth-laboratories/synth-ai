use serde_json::Value;
use std::time::Duration;

use crate::config::{BackendAuth, CoreConfig};
use crate::shared_client::{DEFAULT_CONNECT_TIMEOUT_SECS, DEFAULT_POOL_SIZE};
use crate::CoreError;
use synth_ai_core_types::{CoreEvent, EventPollResponse};

#[derive(Debug, Clone)]
pub enum EventKind {
    PromptLearning,
    Eval,
}

impl EventKind {
    pub fn from_str(kind: &str) -> Option<Self> {
        match kind {
            "prompt_learning" | "prompt-learning" | "promptlearning" => {
                Some(EventKind::PromptLearning)
            }
            "eval" | "evaluation" => Some(EventKind::Eval),
            _ => None,
        }
    }

    fn path(&self, job_id: &str) -> String {
        match self {
            EventKind::PromptLearning => {
                format!("/api/prompt-learning/online/jobs/{job_id}/events")
            }
            EventKind::Eval => format!("/api/eval/jobs/{job_id}/events"),
        }
    }
}

fn event_seq(value: &Value) -> Option<i64> {
    value
        .get("seq")
        .and_then(|v| v.as_i64())
        .or_else(|| value.get("sequence").and_then(|v| v.as_i64()))
}

fn event_type(value: &Value) -> String {
    value
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("event_type").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string()
}

fn event_message(value: &Value) -> Option<String> {
    value
        .get("message")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn event_ts(value: &Value) -> Option<String> {
    value
        .get("ts")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("timestamp").and_then(|v| v.as_str()))
        .or_else(|| value.get("created_at").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
}

fn event_payload(value: &Value) -> Value {
    value.get("data").cloned().unwrap_or_else(|| value.clone())
}

fn to_core_event(value: &Value) -> CoreEvent {
    CoreEvent {
        seq: event_seq(value).unwrap_or(-1),
        event_type: event_type(value),
        message: event_message(value),
        data_json: event_payload(value),
        ts: event_ts(value),
    }
}

fn extract_events(payload: &Value) -> Vec<Value> {
    if let Some(list) = payload.as_array() {
        return list.clone();
    }
    if let Some(events) = payload.get("events").and_then(|v| v.as_array()) {
        return events.clone();
    }
    if let Some(events) = payload.get("data").and_then(|v| v.as_array()) {
        return events.clone();
    }
    Vec::new()
}

fn extract_next_seq(payload: &Value, events: &[CoreEvent]) -> Option<i64> {
    if let Some(next) = payload.get("next_seq").and_then(|v| v.as_i64()) {
        return Some(next);
    }
    if let Some(max_seq) = events
        .iter()
        .filter_map(|e| (e.seq >= 0).then_some(e.seq))
        .max()
    {
        return Some(max_seq + 1);
    }
    None
}

/// Poll events once for a job.
pub async fn poll_events(
    kind: EventKind,
    job_id: &str,
    config: &CoreConfig,
    since_seq: Option<i64>,
    limit: Option<usize>,
) -> Result<EventPollResponse, CoreError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(config.timeout_ms))
        .user_agent(config.user_agent.clone())
        .pool_max_idle_per_host(DEFAULT_POOL_SIZE)
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .build()?;

    let base = config.backend_base_url.trim_end_matches('/');
    let mut url = format!("{base}{}", kind.path(job_id));

    let mut params: Vec<(String, String)> = Vec::new();
    if let Some(since) = since_seq {
        params.push(("since_seq".to_string(), since.to_string()));
    }
    if let Some(limit) = limit {
        params.push(("limit".to_string(), limit.to_string()));
    }
    if !params.is_empty() {
        let query = serde_urlencoded::to_string(params)
            .map_err(|e| CoreError::Protocol(format!("query encode error: {e}")))?;
        url = format!("{url}?{query}");
    }

    let mut req = client.get(url);
    if let Some(api_key) = &config.api_key {
        req = match config.auth {
            BackendAuth::XApiKey => req.header("X-API-Key", api_key),
            BackendAuth::Bearer => req.header("Authorization", format!("Bearer {api_key}")),
        };
    }

    let resp = req.send().await?;
    let payload: Value = resp.json().await?;

    let raw_events = extract_events(&payload);
    let mut events: Vec<CoreEvent> = raw_events.iter().map(to_core_event).collect();

    if let Some(since) = since_seq {
        events.retain(|e| e.seq < 0 || e.seq > since);
    }

    let next_seq = extract_next_seq(&payload, &events);
    let has_more = payload.get("has_more").and_then(|v| v.as_bool());

    Ok(EventPollResponse {
        events,
        next_seq,
        has_more,
    })
}
