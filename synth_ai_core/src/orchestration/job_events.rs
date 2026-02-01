//! OpenResponses-aligned job/candidate event parsing.
//!
//! This mirrors Python's `synth_ai.sdk.shared.orchestration.events.parser`.

use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedJobEvent {
    pub job_id: String,
    pub seq: i64,
    pub ts: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub level: String,
    pub message: String,
    pub data: Value,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub event_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
    #[serde(default)]
    pub schema_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    #[serde(default)]
    pub event_type: Option<String>,
}

impl ValidationResult {
    pub fn success(event_type: Option<String>) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            event_type,
        }
    }

    pub fn failure(errors: Vec<ValidationError>, event_type: Option<String>) -> Self {
        Self {
            valid: false,
            errors,
            event_type,
        }
    }
}

fn event_type_map() -> &'static [(&'static str, &'static str)] {
    &[
        ("job.started", "job.started"),
        ("job.in_progress", "job.in_progress"),
        ("job.completed", "job.completed"),
        ("job.failed", "job.failed"),
        ("job.cancelled", "job.cancelled"),
        ("learning.policy.gepa.job.started", "job.started"),
        ("learning.policy.gepa.job.completed", "job.completed"),
        ("learning.policy.gepa.job.failed", "job.failed"),
        ("learning.policy.mipro.job.started", "job.started"),
        ("learning.policy.mipro.job.completed", "job.completed"),
        ("learning.policy.mipro.job.failed", "job.failed"),
        ("learning.graph.gepa.job.started", "job.started"),
        ("learning.graph.gepa.job.completed", "job.completed"),
        ("learning.graph.gepa.job.failed", "job.failed"),
        ("eval.policy.job.started", "job.started"),
        ("eval.policy.job.completed", "job.completed"),
        ("eval.policy.job.failed", "job.failed"),
        ("eval.verifier.rlm.job.started", "job.started"),
        ("eval.verifier.rlm.job.completed", "job.completed"),
        ("eval.verifier.rlm.job.failed", "job.failed"),
        ("candidate.added", "candidate.added"),
        ("candidate.evaluated", "candidate.evaluated"),
        ("candidate.completed", "candidate.completed"),
        (
            "learning.policy.gepa.candidate.new_best",
            "candidate.evaluated",
        ),
        (
            "learning.policy.gepa.candidate.evaluated",
            "candidate.evaluated",
        ),
        (
            "learning.policy.mipro.candidate.new_best",
            "candidate.evaluated",
        ),
        (
            "learning.graph.gepa.candidate.evaluated",
            "candidate.evaluated",
        ),
    ]
}

fn default_level(event_type: &str) -> &'static str {
    match event_type {
        "job.failed" => "error",
        "job.cancelled" => "warn",
        _ => "info",
    }
}

fn parse_timestamp(value: &Value) -> String {
    let ts = value
        .get("ts")
        .or_else(|| value.get("timestamp"))
        .or_else(|| value.get("created_at"));
    if let Some(s) = ts.and_then(|v| v.as_str()) {
        return s.to_string();
    }
    if let Some(num) = ts.and_then(|v| v.as_f64()) {
        let secs = num.trunc() as i64;
        let nanos = ((num.fract() * 1_000_000_000.0) as u32).min(999_999_999);
        if let chrono::LocalResult::Single(dt) = Utc.timestamp_opt(secs, nanos) {
            return dt.to_rfc3339();
        }
    }
    Utc::now().to_rfc3339()
}

fn map_event_type(raw_type: &str) -> Option<&'static str> {
    for (raw, mapped) in event_type_map() {
        if raw == &raw_type {
            return Some(*mapped);
        }
    }
    for (suffix, mapped) in event_type_map() {
        if raw_type.ends_with(suffix) {
            return Some(*mapped);
        }
    }
    None
}

fn infer_job_status(event_type: &str) -> Option<&'static str> {
    match event_type {
        "job.started" | "job.in_progress" => Some("in_progress"),
        "job.completed" => Some("completed"),
        "job.failed" => Some("failed"),
        "job.cancelled" => Some("cancelled"),
        _ => None,
    }
}

fn infer_candidate_status(event_type: &str, data: &Map<String, Value>) -> Option<String> {
    if let Some(status) = data.get("status").and_then(|v| v.as_str()) {
        return Some(status.to_string());
    }
    match event_type {
        "candidate.added" => Some("in_progress".to_string()),
        "candidate.evaluated" | "candidate.completed" => Some("completed".to_string()),
        _ => None,
    }
}

fn event_kind(event_type: &str) -> &'static str {
    match event_type {
        "candidate.added" | "candidate.evaluated" | "candidate.completed" => "candidate",
        _ => "job",
    }
}

pub fn parse_job_event(raw: &Value, job_id: Option<&str>) -> Option<ParsedJobEvent> {
    let raw_type = raw
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    if raw_type.is_empty() {
        return None;
    }
    let mapped = map_event_type(&raw_type)?;
    let job_id = raw
        .get("job_id")
        .and_then(|v| v.as_str())
        .or(job_id)
        .unwrap_or("")
        .to_string();
    let seq = raw.get("seq").and_then(|v| v.as_i64()).unwrap_or(0);
    let ts = parse_timestamp(raw);
    let level = raw
        .get("level")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| default_level(mapped))
        .to_string();
    let message = raw
        .get("message")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("Event: {}", mapped));
    let data = raw
        .get("data")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let run_id = raw
        .get("run_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let mut candidate_id = None;
    if let Some(map) = data.as_object() {
        if let Some(cid) = map.get("candidate_id").and_then(|v| v.as_str()) {
            candidate_id = Some(cid.to_string());
        } else if let Some(cid) = map.get("version_id").and_then(|v| v.as_str()) {
            candidate_id = Some(cid.to_string());
        }
    }
    if candidate_id.is_none() {
        candidate_id = raw
            .get("candidate_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }
    let status = if event_kind(mapped) == "candidate" {
        data.as_object()
            .and_then(|map| infer_candidate_status(mapped, map))
    } else {
        infer_job_status(mapped).map(|s| s.to_string())
    };

    Some(ParsedJobEvent {
        job_id,
        seq,
        ts,
        event_type: mapped.to_string(),
        level,
        message,
        data,
        run_id,
        status,
        candidate_id,
        event_kind: event_kind(mapped).to_string(),
    })
}

fn validate_required_fields(map: &Map<String, Value>, required: &[&str]) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    for field in required {
        if !map.contains_key(*field) {
            errors.push(ValidationError {
                path: field.to_string(),
                message: format!("Required field '{}' is missing", field),
                schema_path: None,
            });
        }
    }
    errors
}

fn validate_field_type(value: &Value, expected: &str, path: &str) -> Option<ValidationError> {
    let ok = match expected {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() && !value.is_boolean(),
        "number" => (value.as_i64().is_some() || value.as_f64().is_some()) && !value.is_boolean(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    };
    if ok {
        None
    } else {
        Some(ValidationError {
            path: path.to_string(),
            message: format!("Expected {}, got {}", expected, value),
            schema_path: None,
        })
    }
}

fn validate_enum(value: &Value, allowed: &[&str], path: &str) -> Option<ValidationError> {
    if let Some(s) = value.as_str() {
        if allowed.iter().any(|v| *v == s) {
            return None;
        }
        return Some(ValidationError {
            path: path.to_string(),
            message: format!("Value '{}' not in allowed values: {:?}", s, allowed),
            schema_path: None,
        });
    }
    Some(ValidationError {
        path: path.to_string(),
        message: format!("Expected string enum, got {}", value),
        schema_path: None,
    })
}

/// Validate an event against the base job event schema.
pub fn validate_base_event(value: &Value) -> ValidationResult {
    let event_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let map = match value.as_object() {
        Some(m) => m,
        None => {
            return ValidationResult::failure(
                vec![ValidationError {
                    path: "".to_string(),
                    message: "Event must be an object".to_string(),
                    schema_path: None,
                }],
                event_type,
            )
        }
    };

    let mut errors =
        validate_required_fields(map, &["job_id", "seq", "ts", "type", "level", "message"]);
    if !errors.is_empty() {
        return ValidationResult::failure(errors, event_type);
    }

    let type_checks = [
        ("job_id", "string"),
        ("seq", "integer"),
        ("ts", "string"),
        ("type", "string"),
        ("level", "string"),
        ("message", "string"),
    ];
    for (field, expected) in type_checks {
        if let Some(val) = map.get(field) {
            if let Some(err) = validate_field_type(val, expected, field) {
                errors.push(err);
            }
        }
    }

    if let Some(val) = map.get("level") {
        if let Some(err) = validate_enum(val, &["info", "warn", "error"], "level") {
            errors.push(err);
        }
    }

    if let Some(val) = map.get("data") {
        if !val.is_null() {
            if let Some(err) = validate_field_type(val, "object", "data") {
                errors.push(err);
            }
        }
    }

    if let Some(val) = map.get("run_id") {
        if !val.is_null() {
            if let Some(err) = validate_field_type(val, "string", "run_id") {
                errors.push(err);
            }
        }
    }

    if errors.is_empty() {
        ValidationResult::success(event_type)
    } else {
        ValidationResult::failure(errors, event_type)
    }
}
