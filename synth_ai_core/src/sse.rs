//! Server-Sent Events (SSE) streaming helpers.
//!
//! Provides a lightweight wrapper over reqwest + eventsource-stream so SDKs
//! can consume SSE event streams with consistent error handling.

use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::time::Duration;

use crate::errors::CoreError;

/// Parsed SSE event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: String,
    pub retry: Option<std::time::Duration>,
}

/// Stream of SSE events.
pub type SseStream = Pin<Box<dyn Stream<Item = Result<SseEvent, CoreError>> + Send>>;

/// Stream SSE events with default GET + no body.
pub async fn stream_sse(url: String, headers: HeaderMap) -> Result<SseStream, CoreError> {
    stream_sse_request(url, Method::GET, headers, None, None).await
}

/// Stream SSE events with string method (compat wrapper).
pub async fn stream_sse_events(
    url: &str,
    method: &str,
    headers: HeaderMap,
    body: Option<Value>,
    timeout: Option<Duration>,
) -> Result<SseStream, CoreError> {
    let method = Method::from_bytes(method.as_bytes())
        .map_err(|e| CoreError::InvalidInput(format!("invalid HTTP method {}: {}", method, e)))?;
    stream_sse_request(url.to_string(), method, headers, body, timeout).await
}

/// Stream SSE events with full request control.
pub async fn stream_sse_request(
    url: String,
    method: Method,
    headers: HeaderMap,
    json_payload: Option<Value>,
    timeout: Option<Duration>,
) -> Result<SseStream, CoreError> {
    let mut builder = reqwest::Client::builder();
    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }
    let client = builder.build()?;

    let mut req = client.request(method, &url).headers(headers);
    if let Some(payload) = json_payload {
        req = req.json(&payload);
    }

    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(CoreError::http_response(
            status.as_u16(),
            &url,
            "sse_stream_failed",
            Some(&body),
        ));
    }

    let stream = resp.bytes_stream().eventsource().map(|item| match item {
        Ok(evt) => Ok(SseEvent {
            event: evt.event,
            data: evt.data,
            id: evt.id,
            retry: evt.retry,
        }),
        Err(err) => Err(CoreError::Protocol(err.to_string())),
    });

    Ok(Box::pin(stream))
}
