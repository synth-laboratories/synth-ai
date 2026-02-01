use once_cell::sync::Lazy;
use serde_json::{Map, Value};

static EVENT_BASE_SCHEMAS: Lazy<Value> = Lazy::new(|| {
    let raw = include_str!("../../assets/event_base_schemas.json");
    serde_json::from_str(raw).unwrap_or_else(|_| Value::Object(Map::new()))
});

/// Return the full mapping of base event schemas.
pub fn base_event_schemas() -> Value {
    EVENT_BASE_SCHEMAS.clone()
}

/// Return the base job event schema.
pub fn base_job_event_schema() -> Value {
    EVENT_BASE_SCHEMAS
        .get("BASE_JOB_EVENT_SCHEMA")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()))
}

/// Return a named schema if present.
pub fn get_base_schema(name: &str) -> Option<Value> {
    EVENT_BASE_SCHEMAS.get(name).cloned()
}

/// Merge a base schema with an algorithm extension (mirrors Python registry merge logic).
pub fn merge_event_schema(
    base: &Value,
    extension: &Value,
    algorithm: &str,
    event_type: &str,
) -> Value {
    let mut merged = base.clone();

    if let Value::Object(ref mut map) = merged {
        map.insert(
            "$id".to_string(),
            Value::String(format!(
                "https://synth.ai/schemas/events/{}/{}.json",
                algorithm,
                event_type.replace('.', "-")
            )),
        );
        map.insert(
            "title".to_string(),
            Value::String(format!(
                "{}{}Event",
                algorithm.to_uppercase(),
                event_type
                    .replace('.', " ")
                    .split_whitespace()
                    .map(|s| {
                        let mut chars = s.chars();
                        match chars.next() {
                            Some(first) => {
                                first.to_uppercase().collect::<String>() + chars.as_str()
                            }
                            None => String::new(),
                        }
                    })
                    .collect::<String>()
            )),
        );
        if let Value::Object(ext_map) = extension {
            if let Some(desc) = ext_map.get("description") {
                map.insert("description".to_string(), desc.clone());
            }
        }

        if let Some(Value::Array(all_of)) = map.get_mut("allOf") {
            for item in all_of.iter_mut() {
                if let Value::Object(item_map) = item {
                    if let Some(Value::Object(props)) = item_map.get_mut("properties") {
                        if let Some(Value::Object(data_props)) = props.get_mut("data") {
                            let props_entry = data_props
                                .entry("properties".to_string())
                                .or_insert_with(|| Value::Object(Map::new()));
                            if let Value::Object(ref mut data_properties) = props_entry {
                                if let Value::Object(ext_map) = extension {
                                    if let Some(Value::Object(ext_props)) =
                                        ext_map.get("properties")
                                    {
                                        for (key, val) in ext_props {
                                            data_properties.insert(key.clone(), val.clone());
                                        }
                                    }
                                }
                            }

                            if let Value::Object(ext_map) = extension {
                                if let Some(Value::Array(required)) = ext_map.get("required") {
                                    let required_entry = data_props
                                        .entry("required".to_string())
                                        .or_insert_with(|| Value::Array(Vec::new()));
                                    if let Value::Array(ref mut req_list) = required_entry {
                                        for item in required {
                                            if !req_list.contains(item) {
                                                req_list.push(item.clone());
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    merged
}
