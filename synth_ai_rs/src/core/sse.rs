use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::core::http::shared_client;
use crate::types::{Result, SynthError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: String,
    pub retry: Option<std::time::Duration>,
}

pub type SseStream = Pin<Box<dyn Stream<Item = Result<SseEvent>> + Send>>;

pub async fn stream_sse(url: String, headers: HeaderMap) -> Result<SseStream> {
    let resp = shared_client().get(url).headers(headers).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(SynthError::Api {
            status: status.as_u16(),
            body,
        });
    }

    let stream = resp.bytes_stream().eventsource().map(|item| match item {
        Ok(evt) => Ok(SseEvent {
            event: evt.event,
            data: evt.data,
            id: evt.id,
            retry: evt.retry,
        }),
        Err(err) => Err(SynthError::Sse(err.to_string())),
    });

    Ok(Box::pin(stream))
}
