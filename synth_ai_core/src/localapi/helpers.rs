use serde_json::{Map, Value};
use std::collections::HashMap;
use std::env;
use url::Url;

pub fn normalize_chat_completion_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return "/chat/completions".to_string();
    }

    if trimmed.contains("://") {
        if let Ok(mut parsed) = Url::parse(trimmed) {
            let path = parsed.path().trim_end_matches('/');
            if path.ends_with("/v1/chat/completions") || path.ends_with("/chat/completions") {
                return parsed.to_string();
            }

            let new_path =
                if (path.contains("/v1/") && !path.ends_with("/v1")) || path.ends_with("/v1") {
                    format!("{}/chat/completions", path)
                } else if path.ends_with("/completions") {
                    let base = path.rsplitn(2, '/').nth(1).unwrap_or("");
                    if base.is_empty() {
                        "/chat/completions".to_string()
                    } else {
                        format!("{}/chat/completions", base)
                    }
                } else if path.is_empty() {
                    "/v1/chat/completions".to_string()
                } else {
                    format!("{}/v1/chat/completions", path)
                };

            parsed.set_path(&new_path);
            return parsed.to_string();
        }
    }

    let mut base = trimmed.to_string();
    let mut query = String::new();
    let mut fragment = String::new();
    if let Some(idx) = base.find('#') {
        fragment = base[idx..].to_string();
        base = base[..idx].to_string();
    }
    if let Some(idx) = base.find('?') {
        query = base[idx..].to_string();
        base = base[..idx].to_string();
    }

    let path = base.trim_end_matches('/');
    if path.ends_with("/v1/chat/completions") || path.ends_with("/chat/completions") {
        return format!("{}{}{}", path, query, fragment);
    }

    let new_path = if (path.contains("/v1/") && !path.ends_with("/v1")) || path.ends_with("/v1") {
        format!("{}/chat/completions", path)
    } else if path.ends_with("/completions") {
        let base = path.rsplitn(2, '/').nth(1).unwrap_or("");
        if base.is_empty() {
            "/chat/completions".to_string()
        } else {
            format!("{}/chat/completions", base)
        }
    } else if path.is_empty() {
        "/v1/chat/completions".to_string()
    } else {
        format!("{}/v1/chat/completions", path)
    };

    format!("{}{}{}", new_path, query, fragment)
}

pub fn get_default_max_completion_tokens(model_name: &str) -> i32 {
    let lowered = model_name.to_lowercase();
    if lowered.contains("gpt-5") || lowered.contains("gpt5") {
        2048
    } else if lowered.contains("gpt-4") || lowered.contains("gpt4") {
        4096
    } else if lowered.contains("o1") || lowered.contains("o3") {
        16384
    } else if lowered.contains("claude") {
        4096
    } else {
        512
    }
}

pub fn extract_api_key(
    headers: &HashMap<String, String>,
    policy_config: &Value,
    default_env_keys: Option<&HashMap<String, String>>,
) -> Option<String> {
    let mut header_map = HashMap::new();
    for (k, v) in headers {
        header_map.insert(k.to_lowercase(), v.clone());
    }

    let default_map = default_env_keys.cloned().unwrap_or_else(|| {
        let mut map = HashMap::new();
        map.insert("api.groq.com".to_string(), "GROQ_API_KEY".to_string());
        map.insert("api.openai.com".to_string(), "OPENAI_API_KEY".to_string());
        map
    });

    let route_base = policy_config
        .get("inference_url")
        .and_then(|v| v.as_str())
        .or_else(|| policy_config.get("api_base").and_then(|v| v.as_str()))
        .or_else(|| policy_config.get("base_url").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_lowercase();

    for (host, env_var) in default_map {
        if route_base.contains(&host) {
            if let Ok(value) = env::var(env_var) {
                if !value.trim().is_empty() {
                    return Some(value);
                }
            }
        }
    }

    if let Some(value) = header_map.get("x-api-key") {
        if !value.trim().is_empty() {
            return Some(value.clone());
        }
    }

    if let Some(value) = header_map.get("authorization") {
        let trimmed = value.trim();
        if trimmed.to_lowercase().starts_with("bearer ") {
            return Some(trimmed[7..].trim().to_string());
        }
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    None
}

pub fn parse_tool_calls_from_response(
    response_json: &Value,
    expected_tool_name: Option<&str>,
) -> Result<Vec<Value>, String> {
    let mut result = Vec::new();
    let response_obj = match response_json.as_object() {
        Some(obj) => obj,
        None => return Ok(result),
    };
    let choices = response_obj.get("choices").and_then(|v| v.as_array());
    let choices = match choices {
        Some(list) => list,
        None => return Ok(result),
    };
    if choices.is_empty() {
        return Ok(result);
    }
    let empty_map = Map::new();
    let empty_map2 = Map::new();
    let empty_map3 = Map::new();
    let first = choices[0].as_object().unwrap_or(&empty_map);
    let message = first
        .get("message")
        .and_then(|v| v.as_object())
        .unwrap_or(&empty_map2);
    let tool_calls = message.get("tool_calls").and_then(|v| v.as_array());
    let tool_calls = match tool_calls {
        Some(list) => list,
        None => return Ok(result),
    };
    for call in tool_calls {
        let call_obj = match call.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        let function_block = call_obj
            .get("function")
            .and_then(|v| v.as_object())
            .unwrap_or(&empty_map3);
        let name = function_block
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if let Some(expected) = expected_tool_name {
            if !name.is_empty() && name != expected {
                return Err(format!("Unexpected tool name: {}", name));
            }
        }
        let mut function_map = Map::new();
        function_map.insert("name".to_string(), Value::String(name.to_string()));
        let args_value = function_block
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::String("{}".to_string()));
        function_map.insert("arguments".to_string(), args_value);

        let mut tool_call = Map::new();
        tool_call.insert(
            "id".to_string(),
            call_obj
                .get("id")
                .cloned()
                .unwrap_or_else(|| Value::String("".to_string())),
        );
        tool_call.insert(
            "type".to_string(),
            call_obj
                .get("type")
                .cloned()
                .unwrap_or_else(|| Value::String("function".to_string())),
        );
        tool_call.insert("function".to_string(), Value::Object(function_map));
        result.push(Value::Object(tool_call));
    }

    Ok(result)
}
