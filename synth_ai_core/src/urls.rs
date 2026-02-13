use url::Url;

use crate::CoreError;

/// Canonical default URLs for API and infra endpoints.
pub const DEFAULT_BACKEND_URL: &str = "https://api.usesynth.ai";
pub const DEFAULT_FRONTEND_URL: &str = "https://usesynth.ai";
pub const DEFAULT_RUST_BACKEND_URL: &str = "https://infra-api.usesynth.ai";
pub const DEFAULT_DEV_BACKEND_URL: &str = "https://api-dev.usesynth.ai";
pub const DEFAULT_DEV_RUST_BACKEND_URL: &str = "https://infra-api-dev.usesynth.ai";
pub const DEFAULT_LOCAL_RUST_BACKEND_URL: &str = "http://localhost:8080";

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn current_environment() -> String {
    std::env::var("ENVIRONMENT")
        .or_else(|_| std::env::var("APP_ENVIRONMENT"))
        .or_else(|_| std::env::var("RAILWAY_ENVIRONMENT"))
        .or_else(|_| std::env::var("RAILWAY_ENVIRONMENT_NAME"))
        .or_else(|_| std::env::var("ENV"))
        .unwrap_or_else(|_| "dev".to_string())
        .to_lowercase()
}

fn is_prod_environment(env: &str) -> bool {
    matches!(env, "prod" | "production" | "main")
}

fn looks_like_url(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.starts_with("http://") || trimmed.starts_with("https://")
}

fn resolve_backend_url_override() -> Option<String> {
    let raw = std::env::var("SYNTH_BACKEND_URL_OVERRIDE").ok()?;
    let raw = raw.trim().trim_matches('"').trim_matches('\'');
    if raw.is_empty() {
        return None;
    }
    let lowered = raw.to_lowercase();
    match lowered.as_str() {
        "local" | "localhost" => Some(env_or_default("LOCAL_BACKEND_URL", "http://localhost:8000")),
        "dev" | "development" | "staging" | "railway" => Some(env_or_default(
            "DEV_SYNTH_BACKEND_URL",
            &env_or_default("DEV_BACKEND_URL", DEFAULT_DEV_BACKEND_URL),
        )),
        "prod" | "production" | "main" => Some(env_or_default(
            "PROD_SYNTH_BACKEND_URL",
            &env_or_default("PROD_BACKEND_URL", DEFAULT_BACKEND_URL),
        )),
        _ => {
            if looks_like_url(raw) {
                Some(raw.to_string())
            } else {
                None
            }
        }
    }
}

fn resolve_rust_backend_url_override() -> Option<String> {
    let raw = std::env::var("SYNTH_RUST_BACKEND_URL_OVERRIDE").ok()?;
    let raw = raw.trim().trim_matches('"').trim_matches('\'');
    if raw.is_empty() {
        return None;
    }
    let lowered = raw.to_lowercase();
    match lowered.as_str() {
        "local" | "localhost" => Some(env_or_default(
            "LOCAL_RUST_BACKEND_URL",
            &env_or_default("RUST_BACKEND_URL", DEFAULT_LOCAL_RUST_BACKEND_URL),
        )),
        "dev" | "development" | "staging" | "railway" => Some(env_or_default(
            "DEV_RUST_BACKEND_URL",
            &env_or_default("RUST_BACKEND_URL", DEFAULT_DEV_RUST_BACKEND_URL),
        )),
        "prod" | "production" | "main" => Some(env_or_default(
            "PROD_RUST_BACKEND_URL",
            &env_or_default("RUST_BACKEND_URL", DEFAULT_RUST_BACKEND_URL),
        )),
        _ => {
            if looks_like_url(raw) {
                Some(raw.to_string())
            } else {
                None
            }
        }
    }
}

fn resolve_backend_url() -> String {
    if let Some(override_url) = resolve_backend_url_override() {
        return override_url;
    }
    let env = current_environment();
    if is_prod_environment(&env) {
        env_or_default(
            "PROD_SYNTH_BACKEND_URL",
            &env_or_default(
                "SYNTH_BACKEND_URL",
                &env_or_default(
                    "SYNTH_API_URL",
                    &env_or_default(
                        "PROD_BACKEND_URL",
                        &env_or_default("BACKEND_URL", DEFAULT_BACKEND_URL),
                    ),
                ),
            ),
        )
    } else {
        env_or_default(
            "DEV_SYNTH_BACKEND_URL",
            &env_or_default(
                "SYNTH_BACKEND_URL",
                &env_or_default(
                    "SYNTH_API_URL",
                    &env_or_default(
                        "DEV_BACKEND_URL",
                        &env_or_default("BACKEND_URL", "http://localhost:8000"),
                    ),
                ),
            ),
        )
    }
}

fn resolve_rust_backend_url() -> String {
    if let Some(override_url) = resolve_rust_backend_url_override() {
        return override_url;
    }
    let env = current_environment();
    if is_prod_environment(&env) {
        env_or_default(
            "PROD_RUST_BACKEND_URL",
            &env_or_default(
                "SYNTH_RUST_BACKEND_URL",
                &env_or_default("RUST_BACKEND_URL", DEFAULT_RUST_BACKEND_URL),
            ),
        )
    } else {
        env_or_default(
            "DEV_RUST_BACKEND_URL",
            &env_or_default(
                "SYNTH_RUST_BACKEND_URL",
                &env_or_default("RUST_BACKEND_URL", "http://localhost:8080"),
            ),
        )
    }
}

/// Base URL for backend.
pub fn backend_url_base() -> String {
    resolve_backend_url()
}

/// Base URL for frontend.
pub fn frontend_url_base() -> String {
    env_or_default("SYNTH_FRONTEND_URL", DEFAULT_FRONTEND_URL)
}

/// Join base URL with a path.
pub fn join_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if path.is_empty() {
        return base.to_string();
    }
    if path.starts_with('/') {
        format!("{base}{path}")
    } else {
        format!("{base}/{path}")
    }
}

/// Backend API base URL (/api suffix).
pub fn backend_url_api() -> String {
    join_url(&backend_url_base(), "/api")
}

/// Synth Research API base.
pub fn backend_url_synth_research_base() -> String {
    join_url(&backend_url_base(), "/api/synth-research")
}

/// Synth Research OpenAI-compatible base.
pub fn backend_url_synth_research_openai() -> String {
    join_url(&backend_url_synth_research_base(), "/v1")
}

/// Synth Research Anthropic-compatible base.
pub fn backend_url_synth_research_anthropic() -> String {
    backend_url_synth_research_base()
}

/// Local backend URL helper.
pub fn local_backend_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}")
}

/// Backend health URL.
pub fn backend_health_url(base_url: &str) -> String {
    join_url(base_url, "/health")
}

/// Backend /me URL.
pub fn backend_me_url(base_url: &str) -> String {
    join_url(base_url, "/api/v1/me")
}

/// Backend demo keys URL.
pub fn backend_demo_keys_url(base_url: &str) -> String {
    join_url(base_url, "/api/demo/keys")
}

fn strip_terminal_segment<'a>(path: &'a str, segment: &str) -> &'a str {
    let trimmed = path.trim_end_matches('/');
    if trimmed.ends_with(segment) {
        let new_len = trimmed.len().saturating_sub(segment.len());
        return trimmed[..new_len].trim_end_matches('/');
    }
    trimmed
}

/// Normalize backend base URL by stripping /api and /v1 suffixes.
pub fn normalize_backend_base(url: &str) -> Result<Url, CoreError> {
    let mut parsed = Url::parse(url)?;
    let mut path = parsed.path().to_string();
    path = strip_terminal_segment(&path, "/v1").to_string();
    path = strip_terminal_segment(&path, "/api").to_string();
    parsed.set_path(path.trim_end_matches('/'));
    parsed.set_query(None);
    parsed.set_fragment(None);
    Ok(parsed)
}

/// Normalize inference base URL by removing chat/completions suffix.
pub fn normalize_inference_base(url: &str) -> Result<Url, CoreError> {
    let mut parsed = Url::parse(url)?;
    let mut path = parsed.path().trim_end_matches('/').to_string();
    if path.ends_with("/chat/completions") {
        path = strip_terminal_segment(&path, "/chat/completions").to_string();
    } else if path.ends_with("/completions") {
        path = strip_terminal_segment(&path, "/completions").to_string();
    } else if path.ends_with("/chat") {
        path = strip_terminal_segment(&path, "/chat").to_string();
    }
    parsed.set_path(path.trim_end_matches('/'));
    parsed.set_fragment(None);
    Ok(parsed)
}

/// Create a local API URL from host/port.
pub fn make_local_api_url(host: &str, port: u16) -> Result<Url, CoreError> {
    let url = format!("http://{host}:{port}");
    Ok(Url::parse(&url)?)
}

/// Validate that a task app URL is well-formed http(s).
pub fn validate_task_app_url(url: &str) -> Result<Url, CoreError> {
    let parsed = Url::parse(url)?;
    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        other => Err(CoreError::InvalidInput(format!(
            "unsupported scheme: {other}"
        ))),
    }
}
