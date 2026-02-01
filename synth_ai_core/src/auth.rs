//! Authentication utilities for Synth SDK.
//!
//! This module provides:
//! - API key resolution from environment and config files
//! - Credential storage in `~/.synth-ai/user_config.json`
//! - Demo key minting
//! - Device authentication flow (OAuth-style browser auth)

use crate::errors::CoreError;
use crate::shared_client::build_pooled_client;
use crate::utils;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default config directory name
pub const CONFIG_DIR: &str = ".synth-ai";

/// Default config file name
pub const CONFIG_FILE: &str = "user_config.json";
/// Default localapi config file name
pub const LOCALAPI_CONFIG_FILE: &str = "localapi_config.json";

/// Default environment variable for API key
pub const ENV_API_KEY: &str = "SYNTH_API_KEY";

/// Default environment variable for environment API key
pub const ENV_ENVIRONMENT_API_KEY: &str = "ENVIRONMENT_API_KEY";

/// Default frontend URL for device auth
pub const DEFAULT_FRONTEND_URL: &str = "https://usesynth.ai";

/// Default backend URL for API calls
pub const DEFAULT_BACKEND_URL: &str = "https://api.usesynth.ai";

/// Get the default config directory path (~/.synth-ai).
pub fn get_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR)
}

/// Get the default config file path (~/.synth-ai/user_config.json).
pub fn get_config_path() -> PathBuf {
    get_config_dir().join(CONFIG_FILE)
}

/// Get the default localapi config file path (~/.synth-ai/localapi_config.json).
pub fn get_localapi_config_path() -> PathBuf {
    get_config_dir().join(LOCALAPI_CONFIG_FILE)
}

/// Load full user config JSON (not just string credentials).
pub fn load_user_config() -> Result<HashMap<String, Value>, CoreError> {
    let path = get_config_path();
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(&path)
        .map_err(|e| CoreError::Config(format!("failed to read config: {}", e)))?;
    let value: Value = serde_json::from_str(&content)
        .map_err(|e| CoreError::Config(format!("invalid config JSON: {}", e)))?;
    let mut result = HashMap::new();
    if let Value::Object(map) = value {
        for (k, v) in map {
            result.insert(k, v);
        }
    }
    Ok(result)
}

/// Save full user config JSON.
pub fn save_user_config(config: &HashMap<String, Value>) -> Result<(), CoreError> {
    let path = get_config_path();
    let value = Value::Object(config.clone().into_iter().collect());
    utils::write_private_json(&path, &value)
        .map_err(|e| CoreError::Config(format!("failed to write config: {}", e)))?;
    Ok(())
}

/// Update user config with provided values.
pub fn update_user_config(
    updates: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, CoreError> {
    let mut current = load_user_config()?;
    for (k, v) in updates {
        current.insert(k.clone(), v.clone());
    }
    save_user_config(&current)?;
    Ok(current)
}

/// Get API key from environment variable.
///
/// # Arguments
///
/// * `env_key` - Environment variable name (defaults to SYNTH_API_KEY)
pub fn get_api_key_from_env(env_key: Option<&str>) -> Option<String> {
    let key = env_key.unwrap_or(ENV_API_KEY);
    env::var(key).ok().filter(|s| !s.trim().is_empty())
}

/// Load credentials from a JSON config file.
///
/// # Arguments
///
/// * `config_path` - Path to config file (defaults to ~/.synth-ai/user_config.json)
pub fn load_credentials(config_path: Option<&Path>) -> Result<HashMap<String, String>, CoreError> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_config_path);

    if !path.exists() {
        return Ok(HashMap::new());
    }

    let content = fs::read_to_string(&path)
        .map_err(|e| CoreError::Config(format!("failed to read config: {}", e)))?;

    let value: Value = serde_json::from_str(&content)
        .map_err(|e| CoreError::Config(format!("invalid config JSON: {}", e)))?;

    let mut result = HashMap::new();
    if let Value::Object(map) = value {
        for (k, v) in map {
            if let Value::String(s) = v {
                result.insert(k, s);
            }
        }
    }

    Ok(result)
}

/// Store credentials to a JSON config file.
///
/// This creates the parent directory if it doesn't exist and sets
/// restrictive permissions (0600) on the file.
///
/// # Arguments
///
/// * `credentials` - Map of credential names to values
/// * `config_path` - Path to config file (defaults to ~/.synth-ai/user_config.json)
pub fn store_credentials(
    credentials: &HashMap<String, String>,
    config_path: Option<&Path>,
) -> Result<(), CoreError> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_config_path);

    // Create parent directory
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| CoreError::Config(format!("failed to create config dir: {}", e)))?;
    }

    // Load existing config and merge
    let mut existing = load_credentials(Some(&path)).unwrap_or_default();
    for (k, v) in credentials {
        existing.insert(k.clone(), v.clone());
    }

    // Write config
    let content = serde_json::to_string_pretty(&existing)
        .map_err(|e| CoreError::Config(format!("failed to serialize config: {}", e)))?;

    fs::write(&path, content)
        .map_err(|e| CoreError::Config(format!("failed to write config: {}", e)))?;

    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        let _ = fs::set_permissions(&path, perms);
    }

    Ok(())
}

/// Get API key from environment or config file.
///
/// Resolution order:
/// 1. Environment variable (SYNTH_API_KEY by default)
/// 2. Config file (~/.synth-ai/user_config.json)
///
/// # Arguments
///
/// * `env_key` - Environment variable name (defaults to SYNTH_API_KEY)
pub fn get_api_key(env_key: Option<&str>) -> Option<String> {
    // Check environment first
    if let Some(key) = get_api_key_from_env(env_key) {
        return Some(key);
    }

    // Check config file
    if let Ok(creds) = load_credentials(None) {
        let key_name = env_key.unwrap_or(ENV_API_KEY);
        if let Some(key) = creds.get(key_name) {
            if !key.trim().is_empty() {
                return Some(key.clone());
            }
        }
    }

    None
}

/// Device authentication session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAuthSession {
    /// Device code for polling
    pub device_code: String,
    /// URL for user to visit in browser
    pub verification_uri: String,
    /// Session expiration time (Unix timestamp)
    pub expires_at: f64,
}

/// Response from device auth token endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAuthResponse {
    /// Synth API key
    #[serde(default)]
    pub synth_api_key: Option<String>,
    /// Environment API key
    #[serde(default)]
    pub environment_api_key: Option<String>,
    /// Legacy keys format
    #[serde(default)]
    pub keys: Option<HashMap<String, String>>,
}

/// Initialize a device authentication session.
///
/// This starts the OAuth-style device auth flow by requesting a device code
/// from the backend.
///
/// # Arguments
///
/// * `frontend_url` - Frontend URL (defaults to https://usesynth.ai)
pub async fn init_device_auth(frontend_url: Option<&str>) -> Result<DeviceAuthSession, CoreError> {
    let base = frontend_url
        .unwrap_or(DEFAULT_FRONTEND_URL)
        .trim_end_matches('/');
    let url = format!("{}/api/auth/device/init", base);

    let client = build_pooled_client(Some(10));
    let resp =
        client.post(&url).send().await.map_err(|e| {
            CoreError::Authentication(format!("failed to reach init endpoint: {}", e))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(CoreError::Authentication(format!(
            "init failed ({}): {}",
            status,
            body.trim()
        )));
    }

    let data: Value = resp
        .json()
        .await
        .map_err(|e| CoreError::Authentication(format!("invalid JSON response: {}", e)))?;

    let device_code = data["device_code"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::Authentication("missing device_code".to_string()))?
        .to_string();

    let verification_uri = data["verification_uri"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::Authentication("missing verification_uri".to_string()))?
        .to_string();

    let expires_in = data["expires_in"].as_i64().unwrap_or(600);
    let expires_at = now_timestamp() + expires_in as f64;

    Ok(DeviceAuthSession {
        device_code,
        verification_uri,
        expires_at,
    })
}

/// Poll for device auth token.
///
/// This polls the token endpoint until credentials are returned or timeout.
///
/// # Arguments
///
/// * `frontend_url` - Frontend URL (defaults to https://usesynth.ai)
/// * `device_code` - Device code from init
/// * `poll_interval_secs` - Polling interval (default 3 seconds)
/// * `timeout_secs` - Overall timeout (default 600 seconds)
pub async fn poll_device_token(
    frontend_url: Option<&str>,
    device_code: &str,
    poll_interval_secs: Option<u64>,
    timeout_secs: Option<u64>,
) -> Result<HashMap<String, String>, CoreError> {
    let base = frontend_url
        .unwrap_or(DEFAULT_FRONTEND_URL)
        .trim_end_matches('/');
    let url = format!("{}/api/auth/device/token", base);
    let poll_interval = Duration::from_secs(poll_interval_secs.unwrap_or(3));
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(600));
    let start = std::time::Instant::now();

    let client = build_pooled_client(Some(10));

    loop {
        if start.elapsed() >= timeout {
            return Err(CoreError::Timeout(
                "device auth timed out before credentials were returned".to_string(),
            ));
        }

        let resp = client
            .post(&url)
            .json(&serde_json::json!({ "device_code": device_code }))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let data: DeviceAuthResponse = r
                    .json()
                    .await
                    .map_err(|e| CoreError::Authentication(format!("invalid JSON: {}", e)))?;

                return Ok(extract_credentials(data));
            }
            Ok(r) if r.status().as_u16() == 404 || r.status().as_u16() == 410 => {
                return Err(CoreError::Authentication(
                    "device code expired or was revoked".to_string(),
                ));
            }
            _ => {
                // Continue polling
                tokio::time::sleep(poll_interval).await;
            }
        }
    }
}

/// Extract credentials from device auth response, handling legacy format.
fn extract_credentials(data: DeviceAuthResponse) -> HashMap<String, String> {
    let mut result = HashMap::new();

    // Get SYNTH_API_KEY
    let synth_key = data.synth_api_key.filter(|s| !s.is_empty()).or_else(|| {
        data.keys
            .as_ref()
            .and_then(|k| k.get("synth").cloned())
            .filter(|s| !s.is_empty())
    });

    if let Some(key) = synth_key {
        result.insert(ENV_API_KEY.to_string(), key);
    }

    // Get ENVIRONMENT_API_KEY
    let env_key = data
        .environment_api_key
        .filter(|s| !s.is_empty())
        .or_else(|| {
            data.keys.as_ref().and_then(|k| {
                k.get("rl_env")
                    .or_else(|| k.get("environment_api_key"))
                    .cloned()
                    .filter(|s| !s.is_empty())
            })
        });

    if let Some(key) = env_key {
        result.insert(ENV_ENVIRONMENT_API_KEY.to_string(), key);
    }

    result
}

/// Mint a demo API key from the backend.
///
/// # Arguments
///
/// * `backend_url` - Backend URL (defaults to https://api.usesynth.ai)
/// * `ttl_hours` - Key TTL in hours (default 4)
pub async fn mint_demo_key(
    backend_url: Option<&str>,
    ttl_hours: Option<u32>,
) -> Result<String, CoreError> {
    let base = backend_url
        .unwrap_or(DEFAULT_BACKEND_URL)
        .trim_end_matches('/');
    let url = format!("{}/api/demo/keys", base);
    let ttl = ttl_hours.unwrap_or(4);

    let client = build_pooled_client(Some(30));
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "ttl_hours": ttl }))
        .send()
        .await
        .map_err(|e| CoreError::Authentication(format!("failed to mint demo key: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(CoreError::Authentication(format!(
            "demo key minting failed ({}): {}",
            status,
            body.trim()
        )));
    }

    let data: Value = resp
        .json()
        .await
        .map_err(|e| CoreError::Authentication(format!("invalid JSON: {}", e)))?;

    data["api_key"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| CoreError::Authentication("no api_key in response".to_string()))
}

/// Get or mint an API key.
///
/// Resolution order:
/// 1. Environment variable
/// 2. Config file
/// 3. Mint demo key (if allow_mint is true)
///
/// # Arguments
///
/// * `backend_url` - Backend URL for minting
/// * `allow_mint` - Whether to mint a demo key if not found
pub async fn get_or_mint_api_key(
    backend_url: Option<&str>,
    allow_mint: bool,
) -> Result<String, CoreError> {
    // Try environment and config first
    if let Some(key) = get_api_key(None) {
        return Ok(key);
    }

    // Mint if allowed
    if allow_mint {
        return mint_demo_key(backend_url, None).await;
    }

    Err(CoreError::Authentication(
        "SYNTH_API_KEY is required but missing".to_string(),
    ))
}

/// Mask a string for display (shows first 8 chars + "...").
pub fn mask_str(s: &str) -> String {
    if s.len() <= 8 {
        "*".repeat(s.len())
    } else {
        format!("{}...", &s[..8])
    }
}

/// Load credentials from config file and set environment variables.
///
/// This hydrates environment variables from the stored config file,
/// making credentials available to the current process.
///
/// # Returns
///
/// Map of variable names to values that were set.
pub fn load_user_env() -> Result<HashMap<String, String>, CoreError> {
    load_user_env_with(true)
}

/// Load credentials from config and localapi config, setting env vars.
pub fn load_user_env_with(override_env: bool) -> Result<HashMap<String, String>, CoreError> {
    let mut applied: HashMap<String, String> = HashMap::new();

    let mut apply = |mapping: &HashMap<String, Value>| {
        for (k, v) in mapping {
            if v.is_null() {
                continue;
            }
            let value = if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                v.to_string()
            };
            if override_env || env::var(k).is_err() {
                env::set_var(k, &value);
            }
            applied.insert(k.clone(), value);
        }
    };

    let config = load_user_config()?;
    apply(&config);

    // Load localapi config (task app entries)
    let localapi_path = get_localapi_config_path();
    if localapi_path.exists() {
        let raw = fs::read_to_string(&localapi_path)
            .map_err(|e| CoreError::Config(format!("failed to read localapi config: {}", e)))?;
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&raw) {
            if let Some(Value::Object(apps)) = map.get("apps") {
                if let Some(entry) = select_task_app_entry(apps) {
                    if let Some(Value::Object(modal)) = entry.get("modal") {
                        let mut modal_map = HashMap::new();
                        if let Some(v) = modal.get("base_url") {
                            modal_map.insert("TASK_APP_BASE_URL".to_string(), v.clone());
                        }
                        if let Some(v) = modal.get("app_name") {
                            modal_map.insert("TASK_APP_NAME".to_string(), v.clone());
                        }
                        if let Some(v) = modal.get("secret_name") {
                            modal_map.insert("TASK_APP_SECRET_NAME".to_string(), v.clone());
                        }
                        apply(&modal_map);
                    }
                    if let Some(Value::Object(secrets)) = entry.get("secrets") {
                        let mut secrets_map = HashMap::new();
                        if let Some(v) = secrets.get("environment_api_key") {
                            secrets_map.insert("ENVIRONMENT_API_KEY".to_string(), v.clone());
                            secrets_map.insert("DEV_ENVIRONMENT_API_KEY".to_string(), v.clone());
                        }
                        apply(&secrets_map);
                    }
                }
            }
        }
    }

    Ok(applied)
}

fn select_task_app_entry(
    apps: &serde_json::Map<String, Value>,
) -> Option<&serde_json::Map<String, Value>> {
    if apps.is_empty() {
        return None;
    }

    if let Ok(cwd) = env::current_dir() {
        let cwd_str = cwd.to_string_lossy().to_string();
        if let Some(Value::Object(entry)) = apps.get(&cwd_str) {
            return Some(entry);
        }
    }

    let mut best: Option<&serde_json::Map<String, Value>> = None;
    let mut best_ts = String::new();
    for (_key, entry) in apps {
        if let Value::Object(map) = entry {
            let ts = map
                .get("last_used")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if ts > best_ts {
                best_ts = ts;
                best = Some(map);
            }
        }
    }
    best
}

/// Write content to a file atomically using temp file + rename.
///
/// This ensures that the file is either fully written or not at all,
/// preventing partial writes from crashes. Also sets secure permissions
/// on Unix systems.
fn write_atomic(path: &Path, content: &str) -> Result<(), CoreError> {
    use std::io::Write;

    let dir = path
        .parent()
        .ok_or_else(|| CoreError::Config("no parent directory".to_string()))?;

    // Create temp file in same directory for atomic rename
    let mut temp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| CoreError::Config(format!("failed to create temp file: {}", e)))?;

    // Set permissions before writing on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        let _ = temp.as_file().set_permissions(perms);
    }

    // Write content
    temp.write_all(content.as_bytes())
        .map_err(|e| CoreError::Config(format!("failed to write: {}", e)))?;

    // Sync to disk
    temp.as_file()
        .sync_all()
        .map_err(|e| CoreError::Config(format!("failed to sync: {}", e)))?;

    // Atomic rename
    temp.persist(path)
        .map_err(|e| CoreError::Config(format!("failed to persist: {}", e)))?;

    Ok(())
}

/// Store credentials securely using atomic file operations.
///
/// This is an enhanced version of `store_credentials` that uses
/// atomic writes for better security and crash safety.
pub fn store_credentials_atomic(
    credentials: &HashMap<String, String>,
    config_path: Option<&Path>,
) -> Result<(), CoreError> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_config_path);

    // Create parent directory with secure permissions
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| CoreError::Config(format!("failed to create config dir: {}", e)))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o700);
            let _ = fs::set_permissions(parent, perms);
        }
    }

    // Load existing config and merge
    let mut existing = load_credentials(Some(&path)).unwrap_or_default();
    for (k, v) in credentials {
        existing.insert(k.clone(), v.clone());
    }

    // Serialize
    let content = serde_json::to_string_pretty(&existing)
        .map_err(|e| CoreError::Config(format!("failed to serialize: {}", e)))?;

    // Write atomically
    write_atomic(&path, &content)
}

/// Run the interactive device authentication setup.
///
/// This initiates the device auth flow, opens the browser, and
/// waits for the user to authenticate.
///
/// # Arguments
///
/// * `open_browser` - Whether to automatically open the browser
///
/// # Example
///
/// ```ignore
/// // Run setup with automatic browser opening
/// run_setup(true).await?;
/// ```
pub async fn run_setup(open_browser: bool) -> Result<(), CoreError> {
    let session = init_device_auth(None).await?;

    println!("Please visit the following URL to authenticate:");
    println!("  {}", session.verification_uri);

    if open_browser {
        if let Err(_) = open::that(&session.verification_uri) {
            println!("(Could not open browser automatically)");
        }
    }

    println!("\nWaiting for authentication...");

    let creds = poll_device_token(None, &session.device_code, None, None).await?;

    // Store using atomic write
    store_credentials_atomic(&creds, None)?;

    let config_path = get_config_path();
    println!("\nCredentials saved to: {}", config_path.display());
    println!("You can now use the Synth API.");

    Ok(())
}

/// Get current Unix timestamp.
fn now_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_config_path() {
        let path = get_config_path();
        assert!(path.ends_with("user_config.json"));
        assert!(path.to_string_lossy().contains(".synth-ai"));
    }

    #[test]
    fn test_mask_str() {
        assert_eq!(mask_str("short"), "*****");
        assert_eq!(mask_str("sk_live_1234567890"), "sk_live_...");
    }

    #[test]
    fn test_extract_credentials() {
        // Modern format
        let data = DeviceAuthResponse {
            synth_api_key: Some("sk_test_123".to_string()),
            environment_api_key: Some("env_test_456".to_string()),
            keys: None,
        };
        let creds = extract_credentials(data);
        assert_eq!(creds.get("SYNTH_API_KEY"), Some(&"sk_test_123".to_string()));
        assert_eq!(
            creds.get("ENVIRONMENT_API_KEY"),
            Some(&"env_test_456".to_string())
        );

        // Legacy format
        let mut legacy_keys = HashMap::new();
        legacy_keys.insert("synth".to_string(), "sk_legacy".to_string());
        legacy_keys.insert("rl_env".to_string(), "env_legacy".to_string());

        let data = DeviceAuthResponse {
            synth_api_key: None,
            environment_api_key: None,
            keys: Some(legacy_keys),
        };
        let creds = extract_credentials(data);
        assert_eq!(creds.get("SYNTH_API_KEY"), Some(&"sk_legacy".to_string()));
        assert_eq!(
            creds.get("ENVIRONMENT_API_KEY"),
            Some(&"env_legacy".to_string())
        );
    }
}
