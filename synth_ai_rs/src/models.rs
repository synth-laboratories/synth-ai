use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type JsonMap = Map<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmptyResponse {
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRef {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceList<T> {
    #[serde(default)]
    pub items: Vec<T>,
    #[serde(default)]
    pub data: Vec<T>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventsResponse {
    #[serde(default)]
    pub events: Vec<Value>,
    #[serde(default)]
    pub items: Vec<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArtifactsResponse {
    #[serde(default)]
    pub artifacts: Vec<Value>,
    #[serde(default)]
    pub items: Vec<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CursorParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateChangeRequest {
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
