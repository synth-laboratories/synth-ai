//! Graph optimization dataset converters.

use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::errors::CoreError;

const KNOWN_PREFIXES: &[(&str, &str)] = &[
    ("### Instruction", "instruction"),
    ("### Input", "input"),
    ("### Response", "response"),
    ("Instruction:", "instruction"),
    ("Question:", "question"),
    ("Context:", "context"),
    ("Input:", "input"),
    ("Output:", "output"),
    ("Query:", "query"),
    ("Document:", "document"),
    ("Text:", "text"),
    ("Passage:", "passage"),
];

fn push_warning(warnings: &mut Vec<Value>, message: String, example_idx: Option<usize>) {
    let mut map = Map::new();
    map.insert("message".to_string(), Value::String(message));
    if let Some(idx) = example_idx {
        map.insert("example_idx".to_string(), Value::Number(idx.into()));
    }
    warnings.push(Value::Object(map));
}

fn parse_sft_example(example: &Value) -> (Option<String>, Option<String>, Option<String>) {
    let messages = match example.get("messages").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return (None, None, None),
    };
    let mut system = None;
    let mut user = None;
    let mut assistant = None;
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        match role {
            "system" => system = Some(content.to_string()),
            "user" => user = Some(content.to_string()),
            "assistant" => assistant = Some(content.to_string()),
            _ => {}
        }
    }
    (system, user, assistant)
}

fn detect_system_prompt(examples: &[Value]) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut any = false;
    for ex in examples {
        let (system, _, _) = parse_sft_example(ex);
        if let Some(system) = system {
            if !system.is_empty() {
                any = true;
                *counts.entry(system).or_insert(0) += 1;
            }
        }
    }
    if !any {
        return None;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(prompt, _)| prompt)
}

fn infer_template(user_messages: &[String]) -> (Option<String>, Vec<String>) {
    if user_messages.is_empty() {
        return (None, vec!["user_message".to_string()]);
    }
    let sample_len = user_messages.len().min(10);
    let sample = &user_messages[..sample_len];

    let mut detected_fields: Vec<(&str, &str)> = Vec::new();
    for (prefix, field_name) in KNOWN_PREFIXES {
        let matches = sample.iter().filter(|msg| msg.contains(prefix)).count();
        if matches as f64 >= (sample_len as f64 * 0.8) {
            detected_fields.push((*prefix, *field_name));
        }
    }
    if detected_fields.is_empty() {
        return (None, vec!["user_message".to_string()]);
    }

    let first_msg = &sample[0];
    detected_fields.sort_by_key(|(prefix, _)| first_msg.find(prefix).unwrap_or(usize::MAX));

    let mut template_parts = Vec::new();
    let mut field_names = Vec::new();
    for (prefix, field_name) in detected_fields {
        template_parts.push(format!("{prefix} {{{field_name}}}"));
        field_names.push(field_name.to_string());
    }
    (Some(template_parts.join("\n")), field_names)
}

fn extract_fields(user_message: &str, field_names: &[String]) -> Map<String, Value> {
    let mut result: Map<String, Value> = Map::new();

    for field_name in field_names {
        let mut prefix = None;
        for (p, fn_name) in KNOWN_PREFIXES {
            if fn_name == field_name {
                prefix = Some(*p);
                break;
            }
        }
        let prefix = match prefix {
            Some(prefix) => prefix,
            None => continue,
        };
        if !user_message.contains(prefix) {
            continue;
        }
        let mut start = user_message.find(prefix).unwrap_or(0) + prefix.len();
        while start < user_message.len() {
            let c = user_message.as_bytes()[start] as char;
            if !matches!(c, ' ' | ':' | '\n') {
                break;
            }
            start += 1;
        }
        let mut end = user_message.len();
        for (p, _) in KNOWN_PREFIXES {
            if *p != prefix {
                if let Some(pos) = user_message[start..].find(p) {
                    end = end.min(start + pos);
                }
            }
        }
        let value = user_message[start..end].trim();
        result.insert(field_name.to_string(), Value::String(value.to_string()));
    }

    if result.is_empty() {
        result.insert(
            "user_message".to_string(),
            Value::String(user_message.to_string()),
        );
    }
    result
}

fn validate_sft_examples(examples: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut warnings = Vec::new();
    let mut valid = Vec::new();

    for (idx, ex) in examples.iter().enumerate() {
        let messages = match ex.get("messages").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => {
                push_warning(
                    &mut warnings,
                    format!("Missing 'messages' in example {idx}"),
                    Some(idx),
                );
                continue;
            }
        };

        let roles: std::collections::HashSet<&str> = messages
            .iter()
            .filter_map(|m| m.get("role").and_then(|v| v.as_str()))
            .collect();
        if !roles.contains("user") {
            push_warning(
                &mut warnings,
                format!("No 'user' role in example {idx}"),
                Some(idx),
            );
            continue;
        }
        if !roles.contains("assistant") {
            push_warning(
                &mut warnings,
                format!("No 'assistant' role in example {idx}"),
                Some(idx),
            );
            continue;
        }

        let mut assistant_content = None;
        for msg in messages {
            if msg.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                assistant_content = msg.get("content").and_then(|v| v.as_str());
            }
        }
        if assistant_content.unwrap_or("").trim().is_empty() {
            push_warning(
                &mut warnings,
                format!("Empty assistant response in example {idx}"),
                Some(idx),
            );
            continue;
        }

        valid.push(ex.clone());
    }

    (valid, warnings)
}

fn validate_sft_file(path: &std::path::Path) -> Result<(Vec<Value>, Vec<Value>), CoreError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CoreError::Validation(format!("File not found: {} ({})", path.display(), e))
    })?;
    let mut examples = Vec::new();
    let mut warnings = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(value) => examples.push(value),
            Err(_) => {
                push_warning(
                    &mut warnings,
                    format!("Invalid JSON on line {}", idx + 1),
                    Some(idx),
                );
            }
        }
    }

    let (valid, extra_warnings) = validate_sft_examples(&examples);
    warnings.extend(extra_warnings);
    if valid.is_empty() {
        return Err(CoreError::Validation("No valid examples found".to_string()));
    }
    Ok((valid, warnings))
}

pub fn convert_openai_sft(
    source: &Value,
    dataset_name: Option<String>,
    detect_template: bool,
    max_examples: Option<usize>,
) -> Result<Value, CoreError> {
    let dataset_name = dataset_name.unwrap_or_else(|| "converted_sft".to_string());
    let (mut examples, mut warnings) = match source {
        Value::String(path) => validate_sft_file(std::path::Path::new(path))?,
        Value::Array(arr) => {
            let (valid, w) = validate_sft_examples(arr);
            if valid.is_empty() {
                return Err(CoreError::Validation("No valid examples found".to_string()));
            }
            (valid, w)
        }
        _ => {
            return Err(CoreError::Validation(
                "Source must be a path string or list of examples".to_string(),
            ))
        }
    };

    if let Some(max) = max_examples {
        if examples.len() > max {
            examples.truncate(max);
        }
    }

    let system_prompt = detect_system_prompt(&examples);
    let mut system_prompts = std::collections::HashSet::new();
    for ex in &examples {
        if let Some(prompt) = parse_sft_example(ex).0 {
            if !prompt.is_empty() {
                system_prompts.insert(prompt);
            }
        }
    }
    let unique_system_prompts = system_prompts.len();

    let mut parsed: Vec<(Option<String>, Option<String>, Option<String>)> = Vec::new();
    for ex in &examples {
        parsed.push(parse_sft_example(ex));
    }
    let user_messages: Vec<String> = parsed
        .iter()
        .filter_map(|(_, user, _)| user.clone())
        .collect();
    let assistant_messages: Vec<Option<String>> =
        parsed.iter().map(|(_, _, a)| a.clone()).collect();

    let mut template = None;
    let mut field_names = vec!["user_message".to_string()];
    if detect_template && !user_messages.is_empty() {
        let (tmpl, fields) = infer_template(&user_messages);
        template = tmpl;
        field_names = fields;
    }

    let mut tasks = Vec::new();
    let mut gold_outputs = Vec::new();
    for (idx, (parsed_ex, assistant)) in parsed.iter().zip(assistant_messages.iter()).enumerate() {
        let (_, user, _) = parsed_ex;
        let user = match user {
            Some(u) if !u.is_empty() => u,
            _ => continue,
        };
        let assistant = match assistant {
            Some(a) if !a.is_empty() => a,
            _ => continue,
        };
        let task_id = format!("sft_{:04}", idx);
        let input_dict = if template.is_some() && field_names != vec!["user_message".to_string()] {
            Value::Object(extract_fields(user, &field_names))
        } else {
            let mut map = Map::new();
            map.insert("user_message".to_string(), Value::String(user.clone()));
            Value::Object(map)
        };
        let mut task = Map::new();
        task.insert("task_id".to_string(), Value::String(task_id.clone()));
        task.insert("input".to_string(), input_dict);
        tasks.push(Value::Object(task));

        let mut output = Map::new();
        output.insert("response".to_string(), Value::String(assistant.clone()));
        let mut gold = Map::new();
        gold.insert("task_id".to_string(), Value::String(task_id));
        gold.insert("output".to_string(), Value::Object(output));
        gold.insert("score".to_string(), Value::Number(1.into()));
        gold_outputs.push(Value::Object(gold));
    }

    let mut metadata = Map::new();
    metadata.insert("name".to_string(), Value::String(dataset_name));
    metadata.insert(
        "task_description".to_string(),
        Value::String(
            system_prompt
                .clone()
                .unwrap_or_else(|| "Complete the assistant response".to_string()),
        ),
    );
    metadata.insert(
        "source_format".to_string(),
        Value::String("openai_sft".to_string()),
    );

    if let Some(template) = template.clone() {
        metadata.insert("detected_template".to_string(), Value::String(template));
        let mut props = Map::new();
        for field in &field_names {
            let mut prop = Map::new();
            prop.insert("type".to_string(), Value::String("string".to_string()));
            props.insert(field.clone(), Value::Object(prop));
        }
        let mut input_schema = Map::new();
        input_schema.insert("type".to_string(), Value::String("object".to_string()));
        input_schema.insert("properties".to_string(), Value::Object(props));
        metadata.insert("input_schema".to_string(), Value::Object(input_schema));
    }

    let mut stats = Map::new();
    stats.insert(
        "total_examples".to_string(),
        Value::Number((examples.len() as i64).into()),
    );
    stats.insert(
        "skipped_examples".to_string(),
        Value::Number((warnings.len() as i64).into()),
    );
    stats.insert(
        "output_examples".to_string(),
        Value::Number((tasks.len() as i64).into()),
    );
    stats.insert(
        "template_detected".to_string(),
        Value::Bool(template.is_some()),
    );
    stats.insert(
        "detected_fields".to_string(),
        Value::Array(field_names.iter().cloned().map(Value::String).collect()),
    );
    stats.insert(
        "unique_system_prompts".to_string(),
        Value::Number((unique_system_prompts as i64).into()),
    );

    if unique_system_prompts > 1 {
        push_warning(
            &mut warnings,
            format!(
                "Found {} different system prompts; using most common",
                unique_system_prompts
            ),
            None,
        );
    }

    let mut dataset = Map::new();
    dataset.insert("tasks".to_string(), Value::Array(tasks));
    dataset.insert("gold_outputs".to_string(), Value::Array(gold_outputs));
    dataset.insert("metadata".to_string(), Value::Object(metadata));

    let mut result = Map::new();
    result.insert("dataset".to_string(), Value::Object(dataset));
    result.insert("warnings".to_string(), Value::Array(warnings));
    result.insert("stats".to_string(), Value::Object(stats));

    Ok(Value::Object(result))
}
