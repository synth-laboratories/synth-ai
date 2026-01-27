use std::env;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Serialize;
use serde_json::Value;

use crate::types::{Result, SynthError};

const DEFAULT_BASE_URL: &str = "https://api.usesynth.ai";

#[derive(Clone, Copy, Debug)]
pub enum AuthStyle {
    Bearer,
    ApiKey,
    Both,
}

#[derive(Clone)]
pub struct SynthClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl SynthClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let api_key = api_key.into();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            http: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let base_url = env::var("SYNTH_BACKEND_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = env::var("SYNTH_API_KEY").map_err(|_| SynthError::MissingApiKey)?;
        Ok(Self::new(base_url, api_key))
    }

    pub fn api_base(&self) -> String {
        let trimmed = self.base_url.trim_end_matches('/');
        if trimmed.ends_with("/api") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/api")
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub(crate) fn auth_headers(&self, auth: AuthStyle) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if matches!(auth, AuthStyle::Bearer | AuthStyle::Both) {
            let value = format!("Bearer {}", self.api_key);
            if let Ok(hv) = HeaderValue::from_str(&value) {
                headers.insert(AUTHORIZATION, hv);
            }
        }
        if matches!(auth, AuthStyle::ApiKey | AuthStyle::Both) {
            if let Ok(hv) = HeaderValue::from_str(&self.api_key) {
                headers.insert("X-API-Key", hv);
            }
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_string();
        }
        let mut rel = path.trim_start_matches('/');
        if rel.starts_with("api/") {
            rel = &rel[4..];
        }
        format!("{}/{}", self.api_base(), rel)
    }

    pub async fn get_json(&self, path: &str, auth: AuthStyle) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .get(url)
            .headers(self.auth_headers(auth))
            .send()
            .await?;
        Self::json_or_error(resp).await
    }

    pub async fn post_json<T: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
        auth: AuthStyle,
    ) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .http
            .post(url)
            .headers(self.auth_headers(auth))
            .json(body)
            .send()
            .await?;
        Self::json_or_error(resp).await
    }

    pub async fn get_json_fallback(&self, paths: &[&str], auth: AuthStyle) -> Result<Value> {
        let mut last_error = None;
        for path in paths {
            match self.get_json(path, auth).await {
                Ok(val) => return Ok(val),
                Err(err) => {
                    if let SynthError::Api { status, .. } = &err {
                        if *status == 404 {
                            last_error = Some(err);
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            SynthError::UnexpectedResponse("no fallback endpoints succeeded".to_string())
        }))
    }

    pub async fn post_json_fallback<T: Serialize + ?Sized>(
        &self,
        paths: &[&str],
        body: &T,
        auth: AuthStyle,
    ) -> Result<Value> {
        let mut last_error = None;
        for path in paths {
            match self.post_json(path, body, auth).await {
                Ok(val) => return Ok(val),
                Err(err) => {
                    if let SynthError::Api { status, .. } = &err {
                        if *status == 404 {
                            last_error = Some(err);
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            SynthError::UnexpectedResponse("no fallback endpoints succeeded".to_string())
        }))
    }

    async fn json_or_error(resp: reqwest::Response) -> Result<Value> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp.json::<Value>().await?);
        }
        let body = resp.text().await.unwrap_or_default();
        Err(SynthError::Api {
            status: status.as_u16(),
            body,
        })
    }
}
