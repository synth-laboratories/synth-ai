//! Context override types for unified optimization.
//!
//! Context overrides allow modifying task app behavior through file artifacts,
//! environment variables, and preflight scripts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Status of a context override application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationStatus {
    /// Successfully applied.
    Applied,
    /// Partially applied (some components succeeded).
    Partial,
    /// Failed to apply.
    Failed,
    /// Skipped (not applicable).
    Skipped,
}

impl ApplicationStatus {
    /// Returns true if the override was at least partially successful.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Applied | Self::Partial)
    }
}

impl Default for ApplicationStatus {
    fn default() -> Self {
        Self::Applied
    }
}

/// Type of error during override application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationErrorType {
    /// Validation error (invalid content).
    Validation,
    /// Path traversal attempt detected.
    PathTraversal,
    /// Permission denied.
    Permission,
    /// Size limit exceeded.
    SizeLimit,
    /// Operation timed out.
    Timeout,
    /// Runtime error during application.
    Runtime,
    /// Target not found.
    NotFound,
    /// Unknown error.
    Unknown,
}

impl Default for ApplicationErrorType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// A context override containing file artifacts, env vars, and scripts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextOverride {
    /// File artifacts to write (path -> content).
    #[serde(default)]
    pub file_artifacts: HashMap<String, String>,
    /// Preflight script to execute before rollout.
    #[serde(default)]
    pub preflight_script: Option<String>,
    /// Environment variables to set.
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
    /// Type of mutation (e.g., "replace", "patch", "append").
    #[serde(default)]
    pub mutation_type: Option<String>,
    /// When this override was created.
    #[serde(default)]
    pub created_at: Option<String>,
    /// Unique override ID.
    #[serde(default)]
    pub override_id: Option<String>,
    /// Source of the override (e.g., "optimizer", "manual").
    #[serde(default)]
    pub source: Option<String>,
    /// Priority for ordering (higher = applied later).
    #[serde(default)]
    pub priority: Option<i32>,
}

impl ContextOverride {
    /// Create a new empty context override.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if this override is empty.
    pub fn is_empty(&self) -> bool {
        self.file_artifacts.is_empty()
            && self.preflight_script.is_none()
            && self.env_vars.is_empty()
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        let files: usize = self.file_artifacts.values().map(|v| v.len()).sum();
        let script = self.preflight_script.as_ref().map(|s| s.len()).unwrap_or(0);
        let env: usize = self.env_vars.iter().map(|(k, v)| k.len() + v.len()).sum();
        files + script + env
    }

    /// Add a file artifact.
    pub fn with_file(mut self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.file_artifacts.insert(path.into(), content.into());
        self
    }

    /// Set the preflight script.
    pub fn with_preflight_script(mut self, script: impl Into<String>) -> Self {
        self.preflight_script = Some(script.into());
        self
    }

    /// Add an environment variable.
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Set the override ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.override_id = Some(id.into());
        self
    }

    /// Validate the override for security issues.
    pub fn validate(&self) -> Result<(), String> {
        // Check for path traversal in file paths
        for path in self.file_artifacts.keys() {
            if path.contains("..") {
                return Err(format!("Path traversal detected in: {}", path));
            }
            if path.starts_with('/') {
                return Err(format!("Absolute paths not allowed: {}", path));
            }
        }

        // Check for dangerous env var names
        const DANGEROUS_VARS: &[&str] = &["PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"];
        for key in self.env_vars.keys() {
            if DANGEROUS_VARS.contains(&key.as_str()) {
                return Err(format!("Setting {} is not allowed", key));
            }
        }

        Ok(())
    }

    /// Get file count.
    pub fn file_count(&self) -> usize {
        self.file_artifacts.len()
    }

    /// Get env var count.
    pub fn env_var_count(&self) -> usize {
        self.env_vars.len()
    }
}

/// Result of applying a context override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOverrideStatus {
    /// The override ID.
    #[serde(default)]
    pub override_id: Option<String>,
    /// Application status.
    #[serde(rename = "overall_status", alias = "status")]
    pub overall_status: ApplicationStatus,
    /// Error type if failed.
    #[serde(default)]
    pub error_type: Option<ApplicationErrorType>,
    /// Error message if failed.
    #[serde(default)]
    pub error_message: Option<String>,
    /// Files successfully applied.
    #[serde(default)]
    pub files_applied: Vec<String>,
    /// Files that failed.
    #[serde(default)]
    pub files_failed: Vec<String>,
    /// Env vars successfully applied.
    #[serde(default)]
    pub env_vars_applied: Vec<String>,
    /// Whether preflight script succeeded.
    #[serde(default)]
    pub preflight_succeeded: Option<bool>,
    /// Duration in milliseconds.
    #[serde(default)]
    pub duration_ms: Option<i64>,
}

impl ContextOverrideStatus {
    /// Create a success status.
    pub fn success(override_id: Option<String>) -> Self {
        Self {
            override_id,
            overall_status: ApplicationStatus::Applied,
            error_type: None,
            error_message: None,
            files_applied: Vec::new(),
            files_failed: Vec::new(),
            env_vars_applied: Vec::new(),
            preflight_succeeded: None,
            duration_ms: None,
        }
    }

    /// Create a failure status.
    pub fn failure(
        override_id: Option<String>,
        error_type: ApplicationErrorType,
        message: impl Into<String>,
    ) -> Self {
        Self {
            override_id,
            overall_status: ApplicationStatus::Failed,
            error_type: Some(error_type),
            error_message: Some(message.into()),
            files_applied: Vec::new(),
            files_failed: Vec::new(),
            env_vars_applied: Vec::new(),
            preflight_succeeded: None,
            duration_ms: None,
        }
    }

    /// Mark files as applied.
    pub fn with_files_applied(mut self, files: Vec<String>) -> Self {
        self.files_applied = files;
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, ms: i64) -> Self {
        self.duration_ms = Some(ms);
        self
    }
}

impl Default for ContextOverrideStatus {
    fn default() -> Self {
        Self::success(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_override() {
        let override_ = ContextOverride::new();
        assert!(override_.is_empty());
        assert_eq!(override_.size_bytes(), 0);
    }

    #[test]
    fn test_override_with_content() {
        let override_ = ContextOverride::new()
            .with_file("AGENTS.md", "# Agent Instructions\n\nBe helpful.")
            .with_env_var("DEBUG", "true")
            .with_id("override-001");

        assert!(!override_.is_empty());
        assert_eq!(override_.file_count(), 1);
        assert_eq!(override_.env_var_count(), 1);
        assert!(override_.size_bytes() > 0);
    }

    #[test]
    fn test_validation_path_traversal() {
        let override_ = ContextOverride::new().with_file("../etc/passwd", "malicious");

        assert!(override_.validate().is_err());
    }

    #[test]
    fn test_validation_absolute_path() {
        let override_ = ContextOverride::new().with_file("/etc/passwd", "malicious");

        assert!(override_.validate().is_err());
    }

    #[test]
    fn test_validation_dangerous_env() {
        let override_ = ContextOverride::new().with_env_var("PATH", "/malicious");

        assert!(override_.validate().is_err());
    }

    #[test]
    fn test_validation_success() {
        let override_ = ContextOverride::new()
            .with_file("config/settings.json", "{}")
            .with_env_var("MY_VAR", "value");

        assert!(override_.validate().is_ok());
    }

    #[test]
    fn test_status_success() {
        let status = ContextOverrideStatus::success(Some("id-1".to_string()))
            .with_files_applied(vec!["file1.txt".to_string()])
            .with_duration(100);

        assert!(status.overall_status.is_success());
        assert_eq!(status.duration_ms, Some(100));
    }

    #[test]
    fn test_status_failure() {
        let status = ContextOverrideStatus::failure(
            Some("id-1".to_string()),
            ApplicationErrorType::Permission,
            "Access denied",
        );

        assert!(!status.overall_status.is_success());
        assert_eq!(status.error_type, Some(ApplicationErrorType::Permission));
    }

    #[test]
    fn test_serde() {
        let override_ = ContextOverride::new()
            .with_file("test.txt", "content")
            .with_id("test-id");

        let json = serde_json::to_string(&override_).unwrap();
        let parsed: ContextOverride = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.override_id, Some("test-id".to_string()));
        assert_eq!(
            parsed.file_artifacts.get("test.txt"),
            Some(&"content".to_string())
        );
    }
}
