//! Server-Sent Events (SSE) streaming helpers.
//!
//! Provides a lightweight wrapper over reqwest + eventsource-stream so SDKs
//! can consume SSE event streams with consistent error handling.

use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
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

/// Stream of parsed JSON values from SSE events.
pub type JsonSseStream = Pin<Box<dyn Stream<Item = Result<Value, CoreError>> + Send>>;

/// Stream SSE events with default GET + no body.
pub async fn stream_sse(url: String, headers: HeaderMap) -> Result<SseStream, CoreError> {
    stream_sse_request(url, Method::GET, headers, None, None).await
}

/// Stream SSE events with full request control.
pub async fn stream_sse_request(
    url: String,
    method: Method,
    headers: HeaderMap,
    json_payload: Option<Value>,
    timeout: Option<Duration>,
) -> Result<SseStream, CoreError> {
    // NOTE: Do NOT set read_timeout for SSE streams!
    // SSE connections are long-lived and events can be sparse.
    // read_timeout kills the connection if no data arrives within the timeout,
    // even if the connection is healthy (just waiting for server to emit events).
    // Application-level timeouts (in lib.rs) handle stuck streams instead.
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

/// Stream SSE events and parse data as JSON.
///
/// This is a convenience wrapper around `stream_sse_request` that:
/// - Accepts headers as a Vec of (key, value) string tuples
/// - Parses the SSE event data field as JSON
/// - Returns a stream of serde_json::Value
///
/// # Arguments
/// * `url` - The SSE endpoint URL
/// * `method` - HTTP method (e.g., "GET", "POST")
/// * `headers` - Headers as Vec<(String, String)>
/// * `json_payload` - Optional JSON body for POST requests
/// * `timeout` - Optional timeout duration
pub async fn stream_sse_events(
    url: &str,
    method: &str,
    headers: Vec<(String, String)>,
    json_payload: Option<Value>,
    timeout: Option<Duration>,
) -> Result<JsonSseStream, CoreError> {
    // Convert string headers to HeaderMap
    let mut header_map = HeaderMap::new();
    for (key, value) in headers {
        let header_name = HeaderName::try_from(key.as_str())
            .map_err(|e| CoreError::Protocol(format!("Invalid header name '{}': {}", key, e)))?;
        let header_value = HeaderValue::from_str(&value)
            .map_err(|e| CoreError::Protocol(format!("Invalid header value '{}': {}", value, e)))?;
        header_map.insert(header_name, header_value);
    }

    // Parse method string
    let http_method = match method.to_uppercase().as_str() {
        "GET" => Method::GET,
        "POST" => Method::POST,
        "PUT" => Method::PUT,
        "DELETE" => Method::DELETE,
        "PATCH" => Method::PATCH,
        _ => return Err(CoreError::Protocol(format!("Invalid HTTP method: {}", method))),
    };

    // Get the raw SSE stream
    let raw_stream = stream_sse_request(
        url.to_string(),
        http_method,
        header_map,
        json_payload,
        timeout,
    )
    .await?;

    // Transform to parse data as JSON, filtering out empty/heartbeat events
    let json_stream = raw_stream.filter_map(|result| async move {
        match result {
            Err(e) => Some(Err(e)),
            Ok(event) => {
                // Skip empty data (heartbeat/keepalive events)
                if event.data.is_empty() || event.data.trim().is_empty() {
                    return None;
                }
                // Handle [DONE] signal
                if event.data == "[DONE]" {
                    return Some(Ok(Value::String("[DONE]".to_string())));
                }

                // Try to parse as JSON
                let value = match serde_json::from_str::<Value>(&event.data) {
                    Ok(value) => value,
                    Err(_) => {
                        // If not valid JSON, return the raw string as a JSON string value
                        Value::String(event.data)
                    }
                };
                Some(Ok(value))
            }
        }
    });

    Ok(Box::pin(json_stream))
}
