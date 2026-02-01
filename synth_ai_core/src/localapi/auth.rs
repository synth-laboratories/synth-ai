use crate::auth;
use crate::errors::{CoreError, HttpErrorInfo};
use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;
use sodiumoxide::crypto::box_::PublicKey;
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::Duration;
use uuid::Uuid;

pub const ENVIRONMENT_API_KEY_NAME: &str = "ENVIRONMENT_API_KEY";
pub const DEV_ENVIRONMENT_API_KEY_NAME: &str = "DEV_ENVIRONMENT_API_KEY";
pub const ENVIRONMENT_API_KEY_ALIASES_NAME: &str = "ENVIRONMENT_API_KEY_ALIASES";
pub const MAX_ENVIRONMENT_API_KEY_BYTES: usize = 8 * 1024;
pub const SEALED_BOX_ALGORITHM: &str = "libsodium.sealedbox.v1";

const DEV_ENVIRONMENT_API_KEY_NAMES: [&str; 2] =
    ["dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY"];

fn mask(value: &str, prefix: usize) -> String {
    if value.is_empty() {
        return "<empty>".to_string();
    }
    let visible: String = value.chars().take(prefix).collect();
    if value.len() > prefix {
        format!("{visible}...")
    } else {
        visible
    }
}

pub fn normalize_environment_api_key() -> Option<String> {
    if let Ok(value) = env::var(ENVIRONMENT_API_KEY_NAME) {
        if !value.is_empty() {
            return Some(value);
        }
    }
    for env_key in DEV_ENVIRONMENT_API_KEY_NAMES.iter() {
        if let Ok(value) = env::var(env_key) {
            if !value.is_empty() {
                env::set_var(ENVIRONMENT_API_KEY_NAME, &value);
                println!(
                    "[task:auth] {} set from {} (prefix={})",
                    ENVIRONMENT_API_KEY_NAME,
                    env_key,
                    mask(&value, 4)
                );
                return Some(value);
            }
        }
    }
    None
}

pub fn allowed_environment_api_keys() -> Vec<String> {
    let mut keys = HashSet::new();
    if let Some(primary) = normalize_environment_api_key() {
        keys.insert(primary);
    }
    if let Ok(aliases) = env::var(ENVIRONMENT_API_KEY_ALIASES_NAME) {
        for part in aliases.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                keys.insert(trimmed.to_string());
            }
        }
    }
    keys.into_iter().collect()
}

fn extract_candidates(header_values: &[String]) -> Vec<String> {
    let mut candidates = Vec::new();
    for raw in header_values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut value = trimmed.to_string();
        if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
            value = trimmed[7..].trim().to_string();
        }
        for part in value.split(',') {
            let token = part.trim();
            if !token.is_empty() {
                candidates.push(token.to_string());
            }
        }
    }
    candidates
}

pub fn is_api_key_header_authorized(header_values: &[String]) -> bool {
    let allowed = allowed_environment_api_keys();
    if allowed.is_empty() {
        return false;
    }
    let candidates = extract_candidates(header_values);
    if candidates.is_empty() {
        return false;
    }
    let allowed_set: HashSet<String> = allowed.into_iter().collect();
    candidates
        .into_iter()
        .any(|candidate| allowed_set.contains(&candidate))
}

pub fn mint_environment_api_key() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

pub fn encrypt_for_backend(pubkey_b64: &str, secret: &[u8]) -> Result<String, CoreError> {
    if pubkey_b64.trim().is_empty() {
        return Err(CoreError::InvalidInput(
            "public key must be a non-empty base64 string".to_string(),
        ));
    }
    if secret.is_empty() {
        return Err(CoreError::InvalidInput(
            "secret must not be empty".to_string(),
        ));
    }

    let key_bytes = general_purpose::STANDARD
        .decode(pubkey_b64.trim())
        .map_err(|_| CoreError::InvalidInput("public key must be base64-encoded".to_string()))?;
    if key_bytes.len() != 32 {
        return Err(CoreError::InvalidInput(
            "public key must be 32 bytes for X25519".to_string(),
        ));
    }

    if sodiumoxide::init().is_err() {
        return Err(CoreError::Internal("failed to init libsodium".to_string()));
    }

    let pk = PublicKey::from_slice(&key_bytes)
        .ok_or_else(|| CoreError::InvalidInput("invalid public key bytes".to_string()))?;
    let cipher = sodiumoxide::crypto::sealedbox::seal(secret, &pk);
    Ok(general_purpose::STANDARD.encode(cipher))
}

fn normalize_backend_base(backend_base: &str) -> Result<String, CoreError> {
    let mut backend = backend_base.trim().trim_end_matches('/').to_string();
    if backend.ends_with("/api") {
        backend = backend.trim_end_matches("/api").to_string();
    }
    if backend.is_empty() {
        return Err(CoreError::InvalidInput(
            "backend_base must be provided".to_string(),
        ));
    }
    Ok(backend)
}

async fn raise_with_detail(resp: reqwest::Response) -> Result<reqwest::Response, CoreError> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status().as_u16();
    let url = resp.url().to_string();
    let text = resp.text().await.unwrap_or_default();
    let snippet = if text.is_empty() {
        None
    } else {
        Some(text.chars().take(200).collect::<String>())
    };
    Err(CoreError::HttpResponse(HttpErrorInfo {
        status,
        url,
        message: "request failed".to_string(),
        body_snippet: snippet,
    }))
}

pub async fn setup_environment_api_key(
    backend_base: &str,
    synth_api_key: &str,
    token: Option<&str>,
    timeout: f64,
) -> Result<Value, CoreError> {
    let backend = normalize_backend_base(backend_base)?;
    if synth_api_key.trim().is_empty() {
        return Err(CoreError::InvalidInput(
            "synth_api_key must be provided".to_string(),
        ));
    }

    let plaintext = token
        .map(|s| s.to_string())
        .or_else(|| env::var(ENVIRONMENT_API_KEY_NAME).ok())
        .unwrap_or_default();
    if plaintext.trim().is_empty() {
        return Err(CoreError::InvalidInput(
            "ENVIRONMENT_API_KEY must be set (or pass token=...) to upload".to_string(),
        ));
    }
    let token_bytes = plaintext.as_bytes();
    if token_bytes.is_empty() {
        return Err(CoreError::InvalidInput(
            "ENVIRONMENT_API_KEY token must not be empty".to_string(),
        ));
    }
    if token_bytes.len() > MAX_ENVIRONMENT_API_KEY_BYTES {
        return Err(CoreError::InvalidInput(
            "ENVIRONMENT_API_KEY token exceeds 8 KiB limit".to_string(),
        ));
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()
        .map_err(CoreError::Http)?;
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {}", synth_api_key))
            .map_err(|e| CoreError::InvalidInput(e.to_string()))?,
    );

    let pub_url = format!("{}/api/v1/crypto/public-key", backend);
    let resp = client.get(&pub_url).headers(headers.clone()).send().await?;
    let resp = raise_with_detail(resp).await?;
    let doc: Value = resp.json().await.map_err(CoreError::Http)?;
    let public_key = doc
        .get("public_key")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CoreError::InvalidInput("backend response missing public_key".to_string())
        })?;
    if let Some(alg) = doc.get("alg").and_then(|v| v.as_str()) {
        if alg != SEALED_BOX_ALGORITHM {
            return Err(CoreError::InvalidInput(format!(
                "unsupported sealed box algorithm: {alg}"
            )));
        }
    }

    let ciphertext_b64 = encrypt_for_backend(public_key, token_bytes)?;
    let body = serde_json::json!({
        "name": ENVIRONMENT_API_KEY_NAME,
        "ciphertext_b64": ciphertext_b64,
    });
    let post_url = format!("{}/api/v1/env-keys", backend);
    let resp2 = client
        .post(&post_url)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", synth_api_key),
        )
        .json(&body)
        .send()
        .await?;
    let resp2 = raise_with_detail(resp2).await?;
    let upload_doc: Value = resp2
        .json()
        .await
        .unwrap_or(Value::Object(Default::default()));

    let mut result = serde_json::Map::new();
    result.insert("stored".to_string(), Value::Bool(true));
    if let Some(id) = upload_doc.get("id").cloned() {
        result.insert("id".to_string(), id);
    }
    if let Some(name) = upload_doc.get("name").cloned() {
        result.insert("name".to_string(), name);
    }
    if let Some(updated_at) = upload_doc.get("updated_at").cloned() {
        result.insert("updated_at".to_string(), updated_at);
    }
    Ok(Value::Object(result))
}

pub async fn ensure_localapi_auth(
    backend_base: Option<&str>,
    synth_api_key: Option<&str>,
    upload: bool,
    persist: Option<bool>,
) -> Result<String, CoreError> {
    let _ = auth::load_user_env_with(false)?;

    let mut key = env::var(ENVIRONMENT_API_KEY_NAME).unwrap_or_default();
    if key.trim().is_empty() {
        let fallback = env::var(DEV_ENVIRONMENT_API_KEY_NAME).unwrap_or_default();
        if !fallback.trim().is_empty() {
            key = fallback;
            env::set_var(ENVIRONMENT_API_KEY_NAME, &key);
        }
    }

    let mut minted = false;
    if key.trim().is_empty() {
        key = mint_environment_api_key();
        env::set_var(ENVIRONMENT_API_KEY_NAME, &key);
        minted = true;
    }

    env::set_var(DEV_ENVIRONMENT_API_KEY_NAME, &key);

    let persist = persist.unwrap_or_else(|| {
        env::var("SYNTH_LOCALAPI_AUTH_PERSIST")
            .map(|v| v != "0")
            .unwrap_or(true)
    });

    if minted && persist {
        let mut updates = HashMap::new();
        updates.insert(
            ENVIRONMENT_API_KEY_NAME.to_string(),
            Value::String(key.clone()),
        );
        updates.insert(
            DEV_ENVIRONMENT_API_KEY_NAME.to_string(),
            Value::String(key.clone()),
        );
        let _ = auth::update_user_config(&updates)?;
    }

    if upload {
        let synth_key = synth_api_key
            .map(|s| s.to_string())
            .or_else(|| env::var("SYNTH_API_KEY").ok());
        if let (Some(backend), Some(synth_key)) = (backend_base, synth_key.as_deref()) {
            let _ = setup_environment_api_key(backend, synth_key, Some(&key), 15.0).await;
        }
    }

    if key.trim().is_empty() {
        return Err(CoreError::InvalidInput(
            "ENVIRONMENT_API_KEY is required but missing".to_string(),
        ));
    }

    Ok(key)
}
