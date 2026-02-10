//! Artifact types for storing outputs and intermediate results.
//!
//! Artifacts can contain text, structured data, or file references.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Content of an artifact (text or structured data).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ArtifactContent {
    /// Plain text content.
    Text(String),
    /// Structured JSON content.
    Structured(HashMap<String, Value>),
}

impl ArtifactContent {
    /// Create text content.
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Create structured content.
    pub fn structured(data: HashMap<String, Value>) -> Self {
        Self::Structured(data)
    }

    /// Get as text if this is text content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Structured(_) => None,
        }
    }

    /// Get as structured if this is structured content.
    pub fn as_structured(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Self::Text(_) => None,
            Self::Structured(m) => Some(m),
        }
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Text(s) => s.len(),
            Self::Structured(m) => serde_json::to_string(m).map(|s| s.len()).unwrap_or(0),
        }
    }
}

impl Default for ArtifactContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// An artifact produced during a rollout or evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// The artifact content.
    pub content: ArtifactContent,
    /// MIME type (e.g., "text/plain", "application/json", "image/png").
    #[serde(default)]
    pub content_type: Option<String>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Unique artifact ID.
    #[serde(default)]
    pub artifact_id: Option<String>,
    /// Correlation ID linking to a trace.
    #[serde(default)]
    pub trace_correlation_id: Option<String>,
    /// Size in bytes.
    #[serde(default)]
    pub size_bytes: Option<i64>,
    /// SHA-256 hash of content.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Storage location information.
    #[serde(default)]
    pub storage: Option<HashMap<String, Value>>,
    /// When the artifact was created.
    #[serde(default)]
    pub created_at: Option<String>,
    /// Name/label for the artifact.
    #[serde(default)]
    pub name: Option<String>,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
}

impl Artifact {
    /// Create a new text artifact.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: ArtifactContent::text(content),
            content_type: Some("text/plain".to_string()),
            metadata: HashMap::new(),
            artifact_id: None,
            trace_correlation_id: None,
            size_bytes: None,
            sha256: None,
            storage: None,
            created_at: None,
            name: None,
            description: None,
        }
    }

    /// Create a new JSON artifact.
    pub fn json(data: HashMap<String, Value>) -> Self {
        Self {
            content: ArtifactContent::structured(data),
            content_type: Some("application/json".to_string()),
            metadata: HashMap::new(),
            artifact_id: None,
            trace_correlation_id: None,
            size_bytes: None,
            sha256: None,
            storage: None,
            created_at: None,
            name: None,
            description: None,
        }
    }

    /// Set the artifact ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.artifact_id = Some(id.into());
        self
    }

    /// Set the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the trace correlation ID.
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_correlation_id = Some(trace_id.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validate artifact size against a maximum.
    pub fn validate_size(&self, max_size_bytes: i64) -> Result<(), String> {
        let size = self
            .size_bytes
            .unwrap_or_else(|| self.content.size_bytes() as i64);
        if size > max_size_bytes {
            return Err(format!(
                "Artifact size {} bytes exceeds maximum {} bytes",
                size, max_size_bytes
            ));
        }
        Ok(())
    }

    /// Calculate and set size_bytes from content.
    pub fn compute_size(&mut self) {
        self.size_bytes = Some(self.content.size_bytes() as i64);
    }
}

impl Default for Artifact {
    fn default() -> Self {
        Self::text("")
    }
}

/// Collection of artifacts from a rollout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactBundle {
    /// List of artifacts.
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    /// Total size in bytes.
    #[serde(default)]
    pub total_size_bytes: Option<i64>,
    /// Bundle metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl ArtifactBundle {
    /// Create a new empty bundle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an artifact.
    pub fn add(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Get total size of all artifacts.
    pub fn compute_total_size(&mut self) -> i64 {
        let total: i64 = self
            .artifacts
            .iter()
            .map(|a| {
                a.size_bytes
                    .unwrap_or_else(|| a.content.size_bytes() as i64)
            })
            .sum();
        self.total_size_bytes = Some(total);
        total
    }

    /// Get artifact by ID.
    pub fn get_by_id(&self, id: &str) -> Option<&Artifact> {
        self.artifacts
            .iter()
            .find(|a| a.artifact_id.as_deref() == Some(id))
    }

    /// Get artifact by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Artifact> {
        self.artifacts
            .iter()
            .find(|a| a.name.as_deref() == Some(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_artifact() {
        let artifact = Artifact::text("Hello, world!")
            .with_name("greeting")
            .with_id("art-001");

        assert_eq!(artifact.content.as_text(), Some("Hello, world!"));
        assert_eq!(artifact.name, Some("greeting".to_string()));
        assert_eq!(artifact.content_type, Some("text/plain".to_string()));
    }

    #[test]
    fn test_json_artifact() {
        let mut data = HashMap::new();
        data.insert("key".to_string(), serde_json::json!("value"));

        let artifact = Artifact::json(data);

        assert!(artifact.content.as_structured().is_some());
        assert_eq!(artifact.content_type, Some("application/json".to_string()));
    }

    #[test]
    fn test_size_validation() {
        let artifact = Artifact::text("x".repeat(1000));

        assert!(artifact.validate_size(2000).is_ok());
        assert!(artifact.validate_size(500).is_err());
    }

    #[test]
    fn test_artifact_bundle() {
        let mut bundle = ArtifactBundle::new();
        bundle.add(Artifact::text("First").with_name("first"));
        bundle.add(Artifact::text("Second").with_name("second"));

        assert_eq!(bundle.artifacts.len(), 2);
        assert!(bundle.get_by_name("first").is_some());
    }

    #[test]
    fn test_serde() {
        let artifact = Artifact::text("test content").with_id("test-id");

        let json = serde_json::to_string(&artifact).unwrap();
        let parsed: Artifact = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.content.as_text(), Some("test content"));
        assert_eq!(parsed.artifact_id, Some("test-id".to_string()));
    }
}
