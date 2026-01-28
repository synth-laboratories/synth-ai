use thiserror::Error;

pub type Result<T> = std::result::Result<T, SynthError>;

#[derive(Debug, Error)]
pub enum SynthError {
    #[error("missing api key")]
    MissingApiKey,
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
    #[error("unexpected response: {0}")]
    UnexpectedResponse(String),
    #[error("sse error: {0}")]
    Sse(String),
}
