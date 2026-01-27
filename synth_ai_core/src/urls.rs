use url::Url;

use crate::CoreError;

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

