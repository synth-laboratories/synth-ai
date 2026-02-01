use std::collections::HashMap;
use std::env;

const VENDOR_KEYS: [(&str, [&str; 2]); 2] = [
    (
        "OPENAI_API_KEY",
        ["dev_openai_api_key", "DEV_OPENAI_API_KEY"],
    ),
    ("GROQ_API_KEY", ["dev_groq_api_key", "DEV_GROQ_API_KEY"]),
];

fn mask(value: &str, prefix: usize) -> String {
    if value.is_empty() {
        return "<empty>".to_string();
    }
    let visible: String = value.chars().take(prefix).collect();
    if value.len() > prefix {
        format!("{visible}…")
    } else {
        visible
    }
}

pub fn normalize_single(key: &str) -> Option<String> {
    if let Ok(value) = env::var(key) {
        if !value.is_empty() {
            return Some(value);
        }
    }
    let fallbacks = VENDOR_KEYS.iter().find(|(k, _)| *k == key);
    if let Some((_key, fallbacks)) = fallbacks {
        for env_key in fallbacks {
            if let Ok(value) = env::var(env_key) {
                if !value.is_empty() {
                    env::set_var(key, &value);
                    println!(
                        "[task:vendor] {} set from {} (prefix={})",
                        key,
                        env_key,
                        mask(&value, 4)
                    );
                    return Some(value);
                }
            }
        }
    }
    None
}

pub fn normalize_vendor_keys() -> HashMap<String, Option<String>> {
    let mut resolved = HashMap::new();
    for (key, _) in VENDOR_KEYS.iter() {
        resolved.insert((*key).to_string(), normalize_single(key));
    }
    resolved
}

pub fn get_openai_key() -> Option<String> {
    normalize_single("OPENAI_API_KEY")
}

pub fn get_groq_key() -> Option<String> {
    normalize_single("GROQ_API_KEY")
}
