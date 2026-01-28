use futures_util::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use std::pin::Pin;

use synth_ai_core::sse::{stream_sse as core_stream_sse, SseEvent};
use synth_ai_core::CoreError;

use crate::types::{Result, SynthError};

pub type SseStream = Pin<Box<dyn Stream<Item = Result<SseEvent>> + Send>>;

fn map_core_error(err: CoreError) -> SynthError {
    match err {
        CoreError::HttpResponse(info) => SynthError::Api {
            status: info.status,
            body: info.body_snippet.unwrap_or_default(),
        },
        CoreError::Http(err) => SynthError::Http(err),
        CoreError::Protocol(msg) => SynthError::Sse(msg),
        CoreError::Timeout(msg) => SynthError::Sse(msg),
        other => SynthError::UnexpectedResponse(other.to_string()),
    }
}

pub async fn stream_sse(url: String, headers: HeaderMap) -> Result<SseStream> {
    let stream = core_stream_sse(url, headers)
        .await
        .map_err(map_core_error)?;
    let mapped = stream.map(|item| item.map_err(map_core_error));
    Ok(Box::pin(mapped))
}
