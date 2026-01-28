use once_cell::sync::Lazy;
use serde_json::{Map, Value};

static DATA_ENUM_VALUES: Lazy<Value> = Lazy::new(|| {
    let raw = include_str!("../../assets/data_enum_values.json");
    serde_json::from_str(raw).unwrap_or_else(|_| Value::Object(Map::new()))
});

/// Return the full mapping of data enum values.
pub fn data_enum_values() -> Value {
    DATA_ENUM_VALUES.clone()
}
