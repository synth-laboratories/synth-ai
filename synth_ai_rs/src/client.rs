use std::env;

use reqwest::header::HeaderMap;
use serde::Serialize;
use serde_json::Value;

use crate::core::{AuthStyle, CoreClient};
use crate::types::{Result, SynthError};

const DEFAULT_BASE_URL: &str = "https://api.usesynth.ai";

#[derive(Clone)]
pub struct SynthClient {
    core: CoreClient,
}

impl SynthClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            core: CoreClient::new(base_url, api_key),
        }
    }

    pub fn from_env() -> Result<Self> {
        let base_url = env::var("SYNTH_BACKEND_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = env::var("SYNTH_API_KEY").map_err(|_| SynthError::MissingApiKey)?;
        Ok(Self::new(base_url, api_key))
    }

    pub fn api_base(&self) -> String {
        self.core.api_base()
    }

    pub fn base_url(&self) -> &str {
        self.core.base_url()
    }

    pub fn api_key(&self) -> &str {
        self.core.api_key()
    }

    pub fn http(&self) -> &reqwest::Client {
        self.core.http()
    }

    pub(crate) fn auth_headers(&self, auth: AuthStyle) -> HeaderMap {
        self.core.auth_headers(auth)
    }

    pub async fn get_json(&self, path: &str, auth: AuthStyle) -> Result<Value> {
        self.core.get_json(path, auth).await
    }

    pub async fn post_json<T: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
        auth: AuthStyle,
    ) -> Result<Value> {
        self.core.post_json(path, body, auth).await
    }

    pub async fn post_json_with_headers<T: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
        auth: AuthStyle,
        extra_headers: Option<HeaderMap>,
    ) -> Result<Value> {
        self.core
            .post_json_with_headers(path, body, auth, extra_headers)
            .await
    }

    pub async fn get_json_fallback(&self, paths: &[&str], auth: AuthStyle) -> Result<Value> {
        self.core.get_json_fallback(paths, auth).await
    }

    pub async fn post_json_fallback<T: Serialize + ?Sized>(
        &self,
        paths: &[&str],
        body: &T,
        auth: AuthStyle,
    ) -> Result<Value> {
        self.core.post_json_fallback(paths, body, auth).await
    }

    pub async fn post_json_fallback_with_headers<T: Serialize + ?Sized>(
        &self,
        paths: &[&str],
        body: &T,
        auth: AuthStyle,
        extra_headers: Option<HeaderMap>,
    ) -> Result<Value> {
        self.core
            .post_json_fallback_with_headers(paths, body, auth, extra_headers)
            .await
    }
}

pub use crate::core::AuthStyle;
