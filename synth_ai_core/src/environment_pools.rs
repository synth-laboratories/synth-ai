//! Environment Pools client (core).
//!
//! Owns request/response semantics (URL selection, version fallback, headers),
//! so Python/Rust wrappers can stay thin.

use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Method;
use serde_json::Value;
use std::time::Duration;
use url::Url;

use crate::api::SynthClient;
use crate::errors::CoreError;
use crate::http::HttpError;
use crate::sse::{stream_sse_request, SseStream};

/// Client for Synth Environment Pools API.
#[derive(Debug)]
pub struct EnvironmentPoolsClient {
    client: SynthClient,
    api_version: Option<String>,
}

impl EnvironmentPoolsClient {
    /// Create a new Environment Pools client.
    pub fn new(api_key: impl AsRef<str>, base_url: Option<&str>) -> Result<Self, CoreError> {
        let client = SynthClient::new(api_key.as_ref(), base_url)?;
        Ok(Self {
            client,
            api_version: None,
        })
    }

    /// Create a new Environment Pools client with custom request timeout.
    pub fn with_timeout(
        api_key: impl AsRef<str>,
        base_url: Option<&str>,
        timeout_secs: u64,
    ) -> Result<Self, CoreError> {
        let client = SynthClient::with_timeout(api_key.as_ref(), base_url, timeout_secs)?;
        Ok(Self {
            client,
            api_version: None,
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self, CoreError> {
        let client = SynthClient::from_env()?;
        Ok(Self {
            client,
            api_version: None,
        })
    }

    /// Override the API version used for endpoint selection.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Access the underlying Synth API client.
    pub fn core(&self) -> &SynthClient {
        &self.client
    }

    fn resolve_api_version(&self) -> String {
        if let Some(version) = self.api_version.as_ref() {
            if !version.trim().is_empty() {
                return version.clone();
            }
        }
        if let Ok(version) = std::env::var("ENV_POOLS_API_VERSION") {
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

    fn legacy_url(&self, suffix: &str) -> String {
        let base = self.client.base_url().trim_end_matches('/');
        format!(
            "{base}/api/v1/environment-pools/{}",
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

    fn auth_headers(&self) -> Result<HeaderMap, CoreError> {
        let api_key = self.client.http().api_key();
        let mut headers = HeaderMap::new();
        if !api_key.trim().is_empty() {
            let bearer = format!("Bearer {}", api_key);
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&bearer)
                    .map_err(|_| CoreError::InvalidInput("invalid api key".to_string()))?,
            );
            headers.insert(
                "X-API-Key",
                HeaderValue::from_str(api_key)
                    .map_err(|_| CoreError::InvalidInput("invalid api key".to_string()))?,
            );
        }
        headers.insert("Accept", HeaderValue::from_static("text/event-stream"));
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        Ok(headers)
    }

    async fn post_json_with_fallback(
        &self,
        suffix: &str,
        body: &Value,
        idempotency_key: Option<&str>,
    ) -> Result<Value, CoreError> {
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
                    return Err(map_http_error(err));
                }
            }
        }

        Err(map_http_error(
            last_error.unwrap_or_else(|| HttpError::InvalidUrl("no env pools endpoints available".to_string())),
        ))
    }

    async fn get_json_with_fallback(
        &self,
        suffix: &str,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Value, CoreError> {
        let version = self.resolve_api_version();
        let public_url = self.public_url(suffix);
        let legacy_path = self.legacy_path(suffix);
        let attempts = if version == "v1" {
            vec![public_url, legacy_path]
        } else {
            vec![legacy_path, public_url]
        };

        let params_ref = params.as_ref().map(|pairs| {
            pairs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>()
        });

        let mut last_error: Option<HttpError> = None;
        for path in attempts {
            let response = self
                .client
                .http()
                .get_json(&path, params_ref.as_ref().map(|v| v.as_slice()))
                .await;
            match response {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(map_http_error(err));
                }
            }
        }

        Err(map_http_error(
            last_error.unwrap_or_else(|| HttpError::InvalidUrl("no env pools endpoints available".to_string())),
        ))
    }

    async fn get_bytes_with_fallback(
        &self,
        suffix: &str,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Vec<u8>, CoreError> {
        let version = self.resolve_api_version();
        let public_url = self.public_url(suffix);
        let legacy_path = self.legacy_path(suffix);
        let attempts = if version == "v1" {
            vec![public_url, legacy_path]
        } else {
            vec![legacy_path, public_url]
        };

        let params_ref = params.as_ref().map(|pairs| {
            pairs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>()
        });

        let mut last_error: Option<HttpError> = None;
        for path in attempts {
            let response = self
                .client
                .http()
                .get_bytes(&path, params_ref.as_ref().map(|v| v.as_slice()))
                .await;
            match response {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(map_http_error(err));
                }
            }
        }

        Err(map_http_error(
            last_error.unwrap_or_else(|| HttpError::InvalidUrl("no env pools endpoints available".to_string())),
        ))
    }

    async fn put_json_with_fallback(
        &self,
        suffix: &str,
        body: &Value,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Value, CoreError> {
        let version = self.resolve_api_version();
        let public_url = self.public_url(suffix);
        let legacy_path = self.legacy_path(suffix);
        let attempts = if version == "v1" {
            vec![public_url, legacy_path]
        } else {
            vec![legacy_path, public_url]
        };

        let params_ref = params.as_ref().map(|pairs| {
            pairs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>()
        });

        let mut last_error: Option<HttpError> = None;
        for path in attempts {
            let response = self
                .client
                .http()
                .put_json_with_params::<Value>(
                    &path,
                    body,
                    params_ref.as_ref().map(|v| v.as_slice()),
                )
                .await;
            match response {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(map_http_error(err));
                }
            }
        }

        Err(map_http_error(
            last_error.unwrap_or_else(|| {
                HttpError::InvalidUrl("no env pools endpoints available".to_string())
            }),
        ))
    }

    async fn delete_with_fallback(&self, suffix: &str) -> Result<(), CoreError> {
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
            let response = self.client.http().delete(&path).await;
            match response {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if err.status() == Some(404) {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(map_http_error(err));
                }
            }
        }

        Err(map_http_error(
            last_error.unwrap_or_else(|| {
                HttpError::InvalidUrl("no env pools endpoints available".to_string())
            }),
        ))
    }

    // ---------------------------------------------------------------------
    // Public generic request surface (used by wrappers)
    // ---------------------------------------------------------------------

    pub async fn get_json(
        &self,
        suffix: &str,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Value, CoreError> {
        self.get_json_with_fallback(suffix, params).await
    }

    pub async fn post_json(
        &self,
        suffix: &str,
        body: &Value,
        idempotency_key: Option<&str>,
    ) -> Result<Value, CoreError> {
        self.post_json_with_fallback(suffix, body, idempotency_key).await
    }

    pub async fn get_bytes(
        &self,
        suffix: &str,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Vec<u8>, CoreError> {
        self.get_bytes_with_fallback(suffix, params).await
    }

    pub async fn put_json(
        &self,
        suffix: &str,
        body: &Value,
        params: Option<Vec<(String, String)>>,
    ) -> Result<Value, CoreError> {
        self.put_json_with_fallback(suffix, body, params).await
    }

    pub async fn delete(&self, suffix: &str) -> Result<(), CoreError> {
        self.delete_with_fallback(suffix).await
    }

    // ---------------------------------------------------------------------
    // SSE streaming (Environment Pools rollouts)
    // ---------------------------------------------------------------------

    pub async fn stream_rollout_events(
        &self,
        rollout_id: &str,
        since: Option<&str>,
        cursor: Option<&str>,
        limit: Option<u32>,
        timeout: Option<Duration>,
    ) -> Result<SseStream, CoreError> {
        let suffix = format!("rollouts/{}/events", rollout_id);
        let version = self.resolve_api_version();
        let public_url = self.public_url(&suffix);
        let legacy_url = self.legacy_url(&suffix);
        let attempts = if version == "v1" {
            vec![public_url, legacy_url]
        } else {
            vec![legacy_url, public_url]
        };

        let mut headers = self.auth_headers()?;
        if let Some(cursor) = cursor.filter(|value| !value.trim().is_empty()) {
            if let Ok(value) = HeaderValue::from_str(cursor) {
                headers.insert("Last-Event-ID", value);
            }
        }

        let mut last_err: Option<CoreError> = None;
        for base_url in attempts {
            let mut url = Url::parse(&base_url)?;
            {
                let mut qp = url.query_pairs_mut();
                if let Some(since) = since.filter(|value| !value.trim().is_empty()) {
                    qp.append_pair("since", since);
                }
                if let Some(cursor) = cursor.filter(|value| !value.trim().is_empty()) {
                    qp.append_pair("cursor", cursor);
                }
                if let Some(limit) = limit {
                    qp.append_pair("limit", &limit.to_string());
                }
            }

            let stream = stream_sse_request(
                url.to_string(),
                Method::GET,
                headers.clone(),
                None,
                timeout,
            )
            .await;

            match stream {
                Ok(s) => return Ok(s),
                Err(err) => {
                    if err.http_status() == Some(404) {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            CoreError::InvalidInput("no env pools SSE endpoints available".to_string())
        }))
    }
}

fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo::from_http_429(
                    "environment_pools",
                    &detail,
                ))
            } else {
                CoreError::HttpResponse(crate::HttpErrorInfo {
                    status: detail.status,
                    url: detail.url,
                    message: detail.message,
                    body_snippet: detail.body_snippet,
                })
            }
        }
        HttpError::Request(e) => CoreError::Http(e),
        other => CoreError::Internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use futures_util::StreamExt;
    use crate::sse::SseEvent;
    use serde_json::json;

    #[test]
    fn test_legacy_prefix_constant() {
        assert_eq!("/api/v1/environment-pools/", "/api/v1/environment-pools/");
    }

    #[tokio::test]
    async fn test_get_json_falls_back_to_legacy_on_404() {
        let server = MockServer::start();

        let public = server.mock(|when, then| {
            when.method(GET).path("/v1/pools");
            then.status(404);
        });
        let legacy = server.mock(|when, then| {
            when.method(GET).path("/api/v1/environment-pools/pools");
            then.status(200).json_body(json!({"ok": true, "via": "legacy"}));
        });

        let client = EnvironmentPoolsClient::with_timeout("sk_test", Some(&server.base_url()), 5)
            .unwrap();
        let value = client.get_json("pools", None).await.unwrap();
        assert_eq!(value["via"], "legacy");

        public.assert_hits(1);
        legacy.assert_hits(1);
    }

    #[tokio::test]
    async fn test_api_version_legacy_prefers_legacy_endpoint() {
        let server = MockServer::start();

        let legacy = server.mock(|when, then| {
            when.method(GET).path("/api/v1/environment-pools/pools");
            then.status(200).json_body(json!({"ok": true, "via": "legacy"}));
        });
        let public = server.mock(|when, then| {
            when.method(GET).path("/v1/pools");
            then.status(500).json_body(json!({"ok": false}));
        });

        let client = EnvironmentPoolsClient::with_timeout("sk_test", Some(&server.base_url()), 5)
            .unwrap()
            .with_api_version("legacy");
        let value = client.get_json("pools", None).await.unwrap();
        assert_eq!(value["via"], "legacy");

        legacy.assert_hits(1);
        public.assert_hits(0);
    }

    #[tokio::test]
    async fn test_stream_rollout_events_falls_back_to_legacy() {
        let server = MockServer::start();

        let public = server.mock(|when, then| {
            when.method(GET).path("/v1/rollouts/r1/events");
            then.status(404);
        });
        let legacy = server.mock(|when, then| {
            when.method(GET)
                .path("/api/v1/environment-pools/rollouts/r1/events");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body("id: 1\nevent: msg\ndata: {\"ok\":true}\n\n");
        });

        let client = EnvironmentPoolsClient::with_timeout("sk_test", Some(&server.base_url()), 5)
            .unwrap();
        let mut stream = client
            .stream_rollout_events("r1", None, None, None, Some(Duration::from_secs(5)))
            .await
            .unwrap();

        let evt: SseEvent = stream.next().await.unwrap().unwrap();
        assert_eq!(evt.id, "1");
        assert_eq!(evt.event, "msg");
        assert_eq!(evt.data, "{\"ok\":true}");

        public.assert_hits(1);
        legacy.assert_hits(1);
    }

    #[tokio::test]
    async fn test_put_json_sends_query_params() {
        let server = MockServer::start();

        let public = server.mock(|when, then| {
            when.method(PUT)
                .path("/v1/pools/p1")
                .query_param("dry_run", "true");
            then.status(200).json_body(json!({"ok": true}));
        });

        let client = EnvironmentPoolsClient::with_timeout("sk_test", Some(&server.base_url()), 5)
            .unwrap();
        let value = client
            .put_json(
                "pools/p1",
                &json!({"name":"x"}),
                Some(vec![("dry_run".to_string(), "true".to_string())]),
            )
            .await
            .unwrap();
        assert_eq!(value["ok"], true);
        public.assert_hits(1);
    }
}
