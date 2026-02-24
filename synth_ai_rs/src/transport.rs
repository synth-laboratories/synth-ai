use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::types::{Result, SynthError};

#[derive(Clone)]
pub struct Transport {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl Transport {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let http = reqwest::Client::builder()
            .build()
            .map_err(SynthError::Http)?;
        Ok(Self {
            base_url,
            api_key,
            http,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_string();
        }
        format!("{}{}", self.base_url, path)
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.http
            .request(method, self.url(path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
    }

    async fn parse_json<R: DeserializeOwned>(resp: reqwest::Response) -> Result<R> {
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(SynthError::Http)?;
        if status.is_success() {
            if bytes.is_empty() {
                return serde_json::from_str("{}")
                    .map_err(|e| SynthError::UnexpectedResponse(format!("empty JSON response: {e}")));
            }
            serde_json::from_slice::<R>(&bytes).map_err(SynthError::Json)
        } else {
            let body = String::from_utf8_lossy(&bytes).to_string();
            Err(SynthError::Api {
                status: status.as_u16(),
                body,
            })
        }
    }

    pub async fn get_json<R: DeserializeOwned>(&self, path: &str) -> Result<R> {
        let resp = self
            .request(reqwest::Method::GET, path)
            .send()
            .await
            .map_err(SynthError::Http)?;
        Self::parse_json(resp).await
    }

    pub async fn get_json_with_query<Q: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        query: &Q,
    ) -> Result<R> {
        let resp = self
            .request(reqwest::Method::GET, path)
            .query(query)
            .send()
            .await
            .map_err(SynthError::Http)?;
        Self::parse_json(resp).await
    }

    pub async fn post_json<B: Serialize, R: DeserializeOwned>(&self, path: &str, body: &B) -> Result<R> {
        let resp = self
            .request(reqwest::Method::POST, path)
            .json(body)
            .send()
            .await
            .map_err(SynthError::Http)?;
        Self::parse_json(resp).await
    }

    pub async fn patch_json<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let resp = self
            .request(reqwest::Method::PATCH, path)
            .json(body)
            .send()
            .await
            .map_err(SynthError::Http)?;
        Self::parse_json(resp).await
    }

    pub async fn put_json<B: Serialize, R: DeserializeOwned>(&self, path: &str, body: &B) -> Result<R> {
        let resp = self
            .request(reqwest::Method::PUT, path)
            .json(body)
            .send()
            .await
            .map_err(SynthError::Http)?;
        Self::parse_json(resp).await
    }

    pub async fn delete_empty(&self, path: &str) -> Result<()> {
        let resp = self
            .request(reqwest::Method::DELETE, path)
            .send()
            .await
            .map_err(SynthError::Http)?;
        let status = resp.status();
        if status.is_success() {
            return Ok(());
        }
        let body = resp.text().await.unwrap_or_default();
        Err(SynthError::Api {
            status: status.as_u16(),
            body,
        })
    }

    pub async fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let resp = self
            .request(reqwest::Method::GET, path)
            .send()
            .await
            .map_err(SynthError::Http)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(SynthError::Http)?;
        if status.is_success() {
            return Ok(bytes.to_vec());
        }
        Err(SynthError::Api {
            status: status.as_u16(),
            body: String::from_utf8_lossy(&bytes).to_string(),
        })
    }
}
