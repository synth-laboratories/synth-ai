pub fn detect_provider(model_name: Option<&str>) -> String {
    let name = match model_name {
        Some(value) => value.trim().to_lowercase(),
        None => return "unknown".to_string(),
    };
    if name.is_empty() {
        return "unknown".to_string();
    }

    let openai_tokens = [
        "gpt-",
        "text-davinci",
        "text-curie",
        "text-babbage",
        "text-ada",
    ];
    if openai_tokens.iter().any(|token| name.contains(token)) {
        return "openai".to_string();
    }
    if name.contains("claude") || name.contains("anthropic") {
        return "anthropic".to_string();
    }
    if name.contains("palm") || name.contains("gemini") || name.contains("bard") {
        return "google".to_string();
    }
    if name.contains("azure") {
        return "azure".to_string();
    }
    let local_tokens = ["llama", "mistral", "mixtral", "local"];
    if local_tokens.iter().any(|token| name.contains(token)) {
        return "local".to_string();
    }

    "unknown".to_string()
}

pub fn calculate_cost(model_name: &str, input_tokens: i64, output_tokens: i64) -> Option<f64> {
    let name = model_name.trim().to_lowercase();
    if name.is_empty() {
        return None;
    }

    let pricing = [
        ("gpt-4", 0.03_f64, 0.06_f64),
        ("gpt-4-turbo", 0.01_f64, 0.03_f64),
        ("gpt-3.5-turbo", 0.0005_f64, 0.0015_f64),
        ("claude-3-opus", 0.015_f64, 0.075_f64),
        ("claude-3-sonnet", 0.003_f64, 0.015_f64),
        ("claude-3-haiku", 0.00025_f64, 0.00125_f64),
    ];

    for (prefix, input_price, output_price) in pricing.iter() {
        if name.contains(prefix) {
            let input_cost = (input_tokens as f64 / 1000.0) * input_price;
            let output_cost = (output_tokens as f64 / 1000.0) * output_price;
            return Some(input_cost + output_cost);
        }
    }

    None
}
