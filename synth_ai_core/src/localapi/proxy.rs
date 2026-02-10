use regex::Regex;
use serde_json::{json, Map, Value};

const REMOVE_FIELDS: [&str; 4] = [
    "stop_after_tool_calls",
    "thinking_mode",
    "thinking_budget",
    "reasoning",
];

const REMOVE_SAMPLING_FIELDS: [&str; 2] = ["temperature", "top_p"];

const GPT5_MIN_COMPLETION_TOKENS: i64 = 16_000;

const GROQ_JSON_SCHEMA_MODELS: [&str; 4] = [
    "llama-3.3-70b",
    "llama-3.1-70b-versatile",
    "llama-70b",
    "mixtral-8x7b",
];

fn obj_clone(value: &Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    }
}

fn model_is_gpt5(model: Option<&str>) -> bool {
    model.map(|m| m.contains("gpt-5")).unwrap_or(false)
}

pub fn prepare_for_openai(model: Option<&str>, payload: &Value) -> Value {
    let mut sanitized = obj_clone(payload);
    for field in REMOVE_FIELDS.iter() {
        sanitized.remove(*field);
    }

    if model_is_gpt5(model) {
        let max_tokens = sanitized.remove("max_tokens");
        if !sanitized.contains_key("max_completion_tokens") {
            if let Some(Value::Number(num)) = max_tokens.as_ref() {
                if num.is_i64() || num.is_u64() {
                    sanitized.insert("max_completion_tokens".to_string(), max_tokens.unwrap());
                } else if let Some(v) = max_tokens {
                    sanitized.insert("max_completion_tokens".to_string(), v);
                }
            } else if let Some(v) = max_tokens {
                sanitized.insert("max_completion_tokens".to_string(), v);
            }
        } else if let Some(v) = max_tokens {
            if !sanitized.contains_key("max_completion_tokens") {
                sanitized.insert("max_completion_tokens".to_string(), v);
            }
        }

        for field in REMOVE_SAMPLING_FIELDS.iter() {
            sanitized.remove(*field);
        }

        let mct = sanitized
            .get("max_completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if mct < GPT5_MIN_COMPLETION_TOKENS {
            sanitized.insert(
                "max_completion_tokens".to_string(),
                Value::Number(GPT5_MIN_COMPLETION_TOKENS.into()),
            );
        }

        if !sanitized.contains_key("tool_choice") {
            if let Some(Value::Array(tools)) = sanitized.get("tools") {
                if let Some(Value::Object(tool)) = tools.first() {
                    if let Some(Value::Object(func)) = tool.get("function") {
                        if let Some(Value::String(name)) = func.get("name") {
                            sanitized.insert(
                                "tool_choice".to_string(),
                                json!({
                                    "type": "function",
                                    "function": {"name": name},
                                }),
                            );
                        }
                    }
                }
            }
        }

        sanitized.insert("parallel_tool_calls".to_string(), Value::Bool(false));
    }

    Value::Object(sanitized)
}

pub fn normalize_response_format_for_groq(model: Option<&str>, payload: &mut Map<String, Value>) {
    let response_format = payload.get("response_format");
    let response_format = match response_format {
        Some(Value::Object(map)) => map,
        _ => return,
    };

    let model_lower = model.unwrap_or("").to_lowercase();
    let supports_json_schema = GROQ_JSON_SCHEMA_MODELS
        .iter()
        .any(|supported| model_lower.contains(supported));

    if !supports_json_schema {
        if let Some(Value::String(kind)) = response_format.get("type") {
            if kind == "json_schema" {
                payload.insert(
                    "response_format".to_string(),
                    json!({"type": "json_object"}),
                );
            }
        }
    }
}

pub fn prepare_for_groq(model: Option<&str>, payload: &Value) -> Value {
    let mut sanitized = match prepare_for_openai(model, payload) {
        Value::Object(map) => map,
        other => return other,
    };

    let original_has_max_tokens = payload
        .as_object()
        .and_then(|map| map.get("max_tokens"))
        .is_some();

    if !model_is_gpt5(model)
        && sanitized.contains_key("max_completion_tokens")
        && !original_has_max_tokens
    {
        if let Some(value) = sanitized.remove("max_completion_tokens") {
            sanitized.insert("max_tokens".to_string(), value);
        }
    }

    normalize_response_format_for_groq(model, &mut sanitized);
    Value::Object(sanitized)
}

pub fn inject_system_hint(payload: &Value, hint: &str) -> Value {
    if hint.trim().is_empty() {
        return payload.clone();
    }
    let mut cloned = obj_clone(payload);
    if let Some(Value::Array(messages)) = cloned.get_mut("messages") {
        if let Some(Value::Object(first)) = messages.first_mut() {
            if let Some(Value::String(role)) = first.get("role") {
                if role == "system" {
                    if let Some(Value::String(content)) = first.get("content") {
                        if !content.contains(hint) {
                            let mut new_content = content.trim_end().to_string();
                            if !new_content.is_empty() {
                                new_content.push_str("\n\n");
                            }
                            new_content.push_str(hint);
                            first.insert("content".to_string(), Value::String(new_content));
                        }
                    }
                    return Value::Object(cloned);
                }
            }
        }
        let mut system = Map::new();
        system.insert("role".to_string(), Value::String("system".to_string()));
        system.insert("content".to_string(), Value::String(hint.to_string()));
        messages.insert(0, Value::Object(system));
    }
    Value::Object(cloned)
}

pub fn extract_message_text(message: &Value) -> String {
    match message {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .map(extract_message_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => {
            if let Some(Value::String(content)) = map.get("content") {
                return content.clone();
            }
            if let Some(Value::Array(parts)) = map.get("content") {
                return parts
                    .iter()
                    .map(extract_message_text)
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            if let Some(Value::String(text)) = map.get("text") {
                return text.clone();
            }
            String::new()
        }
        other => other.to_string(),
    }
}

fn parse_actions_candidate(candidate: &Value) -> Option<(Vec<String>, String)> {
    if let Value::Object(map) = candidate {
        let mut actions = Vec::new();
        if let Some(value) = map.get("actions") {
            match value {
                Value::Array(list) => {
                    actions = list
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Value::String(s) => {
                    actions = s
                        .split(';')
                        .map(|p| p.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                _ => {}
            }
        }
        let reasoning = map
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if !actions.is_empty() {
            return Some((actions, reasoning));
        }
    }
    None
}

pub fn parse_tool_call_from_text(text: &str) -> (Vec<String>, String) {
    let text = text.trim();
    if text.is_empty() {
        return (Vec::new(), String::new());
    }

    if let Ok(value) = serde_json::from_str::<Value>(text) {
        if let Some((actions, reasoning)) = parse_actions_candidate(&value) {
            return (
                actions,
                if reasoning.is_empty() {
                    text.to_string()
                } else {
                    reasoning
                },
            );
        }
    }

    let json_like = Regex::new(r"\{[^{}]*actions[^{}]*\}").unwrap();
    for cap in json_like.find_iter(text) {
        if let Ok(value) = serde_json::from_str::<Value>(cap.as_str()) {
            if let Some((actions, reasoning)) = parse_actions_candidate(&value) {
                return (
                    actions,
                    if reasoning.is_empty() {
                        text.to_string()
                    } else {
                        reasoning
                    },
                );
            }
        }
    }

    let actions_regex = Regex::new(r"(?i)actions?\s*:\s*([^\n]+)").unwrap();
    if let Some(cap) = actions_regex.captures(text) {
        if let Some(m) = cap.get(1) {
            let items = m
                .as_str()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            if !items.is_empty() {
                let reasoning = text[..cap.get(0).unwrap().start()].trim().to_string();
                return (items, reasoning);
            }
        }
    }

    let mut actions = Vec::new();
    let mut reasoning_lines = Vec::new();
    let action_line = Regex::new(r"(?i)^action\s*\d*\s*[:\-]\s*(.+)$").unwrap();
    for line in text.lines() {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        if let Some(cap) = action_line.captures(stripped) {
            if let Some(m) = cap.get(1) {
                let candidate = m.as_str().trim().to_string();
                if !candidate.is_empty() {
                    actions.push(candidate);
                }
            }
        } else {
            reasoning_lines.push(stripped.to_string());
        }
    }
    if !actions.is_empty() {
        return (actions, reasoning_lines.join("\n").trim().to_string());
    }

    (Vec::new(), text.to_string())
}

pub fn synthesize_tool_call_if_missing(openai_response: &Value, fallback_tool_name: &str) -> Value {
    let mut response = match openai_response {
        Value::Object(map) => map.clone(),
        _ => return openai_response.clone(),
    };

    let choices = match response.get("choices") {
        Some(Value::Array(list)) if !list.is_empty() => list.clone(),
        _ => return openai_response.clone(),
    };

    let first = match choices.first() {
        Some(Value::Object(map)) => map.clone(),
        _ => return openai_response.clone(),
    };

    let message = match first.get("message") {
        Some(Value::Object(map)) => map.clone(),
        _ => return openai_response.clone(),
    };

    if let Some(Value::Array(tool_calls)) = message.get("tool_calls") {
        if !tool_calls.is_empty() {
            return openai_response.clone();
        }
    }

    let text = extract_message_text(&Value::Object(message.clone()));
    let (actions, reasoning) = parse_tool_call_from_text(&text);
    if actions.is_empty() {
        return openai_response.clone();
    }

    let mut payload = Map::new();
    let actions_value = actions
        .iter()
        .map(|a| Value::String(a.trim().to_string()))
        .filter(|s| !s.as_str().unwrap_or("").is_empty())
        .collect::<Vec<_>>();
    payload.insert("actions".to_string(), Value::Array(actions_value));
    if !reasoning.trim().is_empty() {
        payload.insert(
            "reasoning".to_string(),
            Value::String(reasoning.trim().to_string()),
        );
    }

    let tool_call = json!({
        "id": format!("tool_{}_fallback", fallback_tool_name),
        "type": "function",
        "function": {
            "name": fallback_tool_name,
            "arguments": serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string()),
        },
    });

    let mut new_message = message.clone();
    new_message.insert("tool_calls".to_string(), Value::Array(vec![tool_call]));
    if !new_message.contains_key("content") {
        new_message.insert("content".to_string(), Value::Null);
    }

    let mut new_first = first.clone();
    new_first.insert("message".to_string(), Value::Object(new_message));

    let mut new_choices = vec![Value::Object(new_first)];
    for choice in choices.into_iter().skip(1) {
        new_choices.push(choice);
    }
    response.insert("choices".to_string(), Value::Array(new_choices));
    Value::Object(response)
}
