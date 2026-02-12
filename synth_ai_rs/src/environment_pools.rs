use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Map, Value};
use std::env;

use synth_ai_core::http::HttpError;

use crate::{Error, Result};

/// Client for Synth Environment Pools rollouts.
#[derive(Debug)]
pub struct EnvironmentPoolsClient {
    client: synth_ai_core::SynthClient,
    api_version: Option<String>,
}

impl EnvironmentPoolsClient {
    /// Create a new Environment Pools client.
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>) -> Result<Self> {
        let api_key = api_key.into();
        let client = synth_ai_core::SynthClient::new(&api_key, base_url).map_err(Error::Core)?;
        Ok(Self {
            client,
            api_version: None,
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("SYNTH_API_KEY").map_err(|_| Error::MissingApiKey)?;
        let base_url = env::var("SYNTH_BACKEND_URL").ok();
        Self::new(api_key, base_url.as_deref())
    }

    /// Override the API version used for endpoints.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Access the underlying core client.
    pub fn core(&self) -> &synth_ai_core::SynthClient {
        &self.client
    }

    fn resolve_api_version(&self) -> String {
        if let Some(version) = self.api_version.as_ref() {
            if !version.trim().is_empty() {
                return version.clone();
            }
        }
        if let Ok(version) = env::var("ENV_POOLS_API_VERSION") {
            if !version.trim().is_empty() {
                return version;
            }
        }
        "v1".to_string()
    }

    fn public_url(&self, suffix: &str) -> String {
        let base = self.client.base_url().trim_end_matches('/');
        format!("{base}/v1/{}", suffix.trim_start_matches('/'))
    }

    fn legacy_path(&self, suffix: &str) -> String {
        format!(
            "/api/v1/environment-pools/{}",
            suffix.trim_start_matches('/')
        )
    }

    fn idempotency_headers(idempotency_key: Option<&str>) -> Option<HeaderMap> {
        let key = idempotency_key.filter(|value| !value.trim().is_empty())?;
        let mut headers = HeaderMap::new();
        if let Ok(value) = HeaderValue::from_str(key) {
            headers.insert("Idempotency-Key", value);
        }
        Some(headers)
    }

    async fn post_json_with_fallback(
        &self,
        suffix: &str,
        body: &Value,
        idempotency_key: Option<&str>,
    ) -> Result<Value> {
        let version = self.resolve_api_version();
        let public_url = self.public_url(suffix);
        let legacy_path = self.legacy_path(suffix);
        let headers = Self::idempotency_headers(idempotency_key);
        let attempts = if version == "v1" {
            vec![public_url, legacy_path]
        } else {
            vec![legacy_path, public_url]
        };

        let mut last_error: Option<HttpError> = None;
        for path in attempts {
            let response = self
                .client
                .http()
                .post_json_with_headers::<Value>(&path, body, headers.clone())
                .await;
            match response {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(Error::Core(err.into()));
                }
            }
        }

        Err(Error::Core(
            last_error
                .unwrap_or_else(|| {
                    HttpError::InvalidUrl("no env pools endpoints available".to_string())
                })
                .into(),
        ))
    }

    async fn get_json_with_fallback(&self, suffix: &str) -> Result<Value> {
        let version = self.resolve_api_version();
        let public_url = self.public_url(suffix);
        let legacy_path = self.legacy_path(suffix);
        let attempts = if version == "v1" {
            vec![public_url, legacy_path]
        } else {
            vec![legacy_path, public_url]
        };

        let mut last_error: Option<HttpError> = None;
        for path in attempts {
            let response = self.client.http().get_json(&path, None).await;
            match response {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(Error::Core(err.into()));
                }
            }
        }

        Err(Error::Core(
            last_error
                .unwrap_or_else(|| {
                    HttpError::InvalidUrl("no env pools endpoints available".to_string())
                })
                .into(),
        ))
    }

    /// Create a rollout in Environment Pools.
    pub async fn create_rollout(
        &self,
        mut request: Value,
        idempotency_key: Option<&str>,
        dry_run: Option<bool>,
    ) -> Result<Value> {
        if let Some(dry_run) = dry_run {
            if let Value::Object(map) = &mut request {
                map.insert("dry_run".to_string(), Value::Bool(dry_run));
            }
        }
        self.post_json_with_fallback("rollouts", &request, idempotency_key)
            .await
    }

    /// Create a batch of rollouts.
    pub async fn create_rollouts_batch(
        &self,
        requests: Vec<Value>,
        metadata: Option<Value>,
        idempotency_key: Option<&str>,
    ) -> Result<Value> {
        let mut payload = Map::new();
        payload.insert("requests".to_string(), Value::Array(requests));
        if let Some(metadata) = metadata {
            payload.insert("metadata".to_string(), metadata);
        }
        self.post_json_with_fallback("rollouts/batch", &Value::Object(payload), idempotency_key)
            .await
    }

    /// Fetch rollout status/details.
    pub async fn get_rollout(&self, rollout_id: &str) -> Result<Value> {
        let suffix = format!("rollouts/{}", rollout_id);
        self.get_json_with_fallback(&suffix).await
    }
}
