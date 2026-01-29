use once_cell::sync::Lazy;
use serde_json::{Map, Value};

use crate::CoreError;

static EVENT_ENUM_VALUES: Lazy<Value> = Lazy::new(|| {
    let raw = include_str!("../../assets/event_enum_values.json");
    serde_json::from_str(raw).unwrap_or_else(|_| Value::Object(Map::new()))
});

/// Return the full mapping of orchestration enum values.
pub fn event_enum_values() -> Value {
    EVENT_ENUM_VALUES.clone()
}

fn event_type_values() -> Option<Vec<String>> {
    let map = EVENT_ENUM_VALUES.get("EventType")?.as_object()?;
    Some(
        map.values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
    )
}

/// Check if an event type string is a known event type.
pub fn is_valid_event_type(event_type: &str) -> bool {
    let Some(values) = event_type_values() else {
        return false;
    };
    values.iter().any(|v| v == event_type)
}

/// Validate and return the event type string.
pub fn validate_event_type(event_type: &str) -> Result<String, CoreError> {
    if is_valid_event_type(event_type) {
        Ok(event_type.to_string())
    } else {
        Err(CoreError::Validation(format!(
            "Unknown event type: {:?}. Use EventType enum or add new type to events/types.py",
            event_type
        )))
    }
}
