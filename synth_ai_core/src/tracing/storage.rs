//! Trace storage trait definition.
//!
//! This module defines the abstract `TraceStorage` trait that storage
//! backends must implement.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

use super::error::TracingError;
use super::models::{
    EventReward, MarkovBlanketMessage, OutcomeReward, SessionTimeStep, SessionTrace, TracingEvent,
};

/// Abstract storage backend for traces.
///
/// Implementations can store traces in SQLite, Turso, or other backends.
#[async_trait]
pub trait TraceStorage: Send + Sync {
    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    /// Initialize the storage backend (create tables, etc.).
    async fn initialize(&self) -> Result<(), TracingError>;

    /// Close the storage connection.
    async fn close(&self) -> Result<(), TracingError>;

    // ========================================================================
    // SESSION OPERATIONS
    // ========================================================================

    /// Ensure a session exists (create if not present).
    ///
    /// This is idempotent - calling multiple times with the same session_id
    /// will not create duplicates.
    async fn ensure_session(
        &self,
        session_id: &str,
        created_at: DateTime<Utc>,
        metadata: &Value,
    ) -> Result<(), TracingError>;

    /// Get a session trace by ID.
    async fn get_session(&self, session_id: &str) -> Result<Option<SessionTrace>, TracingError>;

    /// Delete a session and all related data.
    async fn delete_session(&self, session_id: &str) -> Result<bool, TracingError>;

    // ========================================================================
    // TIMESTEP OPERATIONS
    // ========================================================================

    /// Ensure a timestep exists and return its database ID.
    ///
    /// This is idempotent - calling multiple times with the same session_id
    /// and step_id will return the same database ID.
    async fn ensure_timestep(
        &self,
        session_id: &str,
        step: &SessionTimeStep,
    ) -> Result<i64, TracingError>;

    /// Update a timestep (e.g., mark as completed).
    async fn update_timestep(
        &self,
        session_id: &str,
        step_id: &str,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), TracingError>;

    // ========================================================================
    // EVENT OPERATIONS
    // ========================================================================

    /// Insert an event and return its database ID.
    async fn insert_event(
        &self,
        session_id: &str,
        timestep_db_id: Option<i64>,
        event: &TracingEvent,
    ) -> Result<i64, TracingError>;

    // ========================================================================
    // MESSAGE OPERATIONS
    // ========================================================================

    /// Insert a message and return its database ID.
    async fn insert_message(
        &self,
        session_id: &str,
        timestep_db_id: Option<i64>,
        msg: &MarkovBlanketMessage,
    ) -> Result<i64, TracingError>;

    // ========================================================================
    // REWARD OPERATIONS
    // ========================================================================

    /// Insert an outcome (session-level) reward.
    async fn insert_outcome_reward(
        &self,
        session_id: &str,
        reward: &OutcomeReward,
    ) -> Result<i64, TracingError>;

    /// Insert an event-level reward.
    async fn insert_event_reward(
        &self,
        session_id: &str,
        event_id: i64,
        message_id: Option<i64>,
        turn_number: Option<i32>,
        reward: &EventReward,
    ) -> Result<i64, TracingError>;

    // ========================================================================
    // QUERY OPERATIONS
    // ========================================================================

    /// Execute a raw SQL query and return results as JSON values.
    async fn query(&self, sql: &str, params: QueryParams) -> Result<Vec<Value>, TracingError>;

    /// Update session counts (num_timesteps, num_events, num_messages).
    async fn update_session_counts(&self, session_id: &str) -> Result<(), TracingError>;
}

/// Query parameters for trace storage queries.
#[derive(Debug, Clone, Default)]
pub enum QueryParams {
    /// No parameters provided.
    #[default]
    None,
    /// Positional parameters (e.g., ?1, ?2).
    Positional(Vec<Value>),
    /// Named parameters (e.g., :session_id).
    Named(Vec<(String, Value)>),
}

impl QueryParams {
    /// Returns true if there are no parameters.
    pub fn is_empty(&self) -> bool {
        matches!(self, QueryParams::None)
    }
}

impl From<Vec<Value>> for QueryParams {
    fn from(values: Vec<Value>) -> Self {
        if values.is_empty() {
            QueryParams::None
        } else {
            QueryParams::Positional(values)
        }
    }
}

impl From<Vec<(String, Value)>> for QueryParams {
    fn from(values: Vec<(String, Value)>) -> Self {
        if values.is_empty() {
            QueryParams::None
        } else {
            QueryParams::Named(values)
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Database URL (file path, :memory:, or libsql:// URL)
    pub db_url: String,
    /// Auth token for remote databases
    pub auth_token: Option<String>,
    /// Whether to auto-initialize schema
    pub auto_init: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_url: ":memory:".to_string(),
            auth_token: None,
            auto_init: true,
        }
    }
}

impl StorageConfig {
    /// Create config for in-memory database.
    pub fn memory() -> Self {
        Self::default()
    }

    /// Create config for a file-based database.
    pub fn file(path: impl Into<String>) -> Self {
        Self {
            db_url: path.into(),
            auth_token: None,
            auto_init: true,
        }
    }

    /// Create config for a remote Turso database.
    pub fn turso(url: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            db_url: url.into(),
            auth_token: Some(token.into()),
            auto_init: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.db_url, ":memory:");
        assert!(config.auth_token.is_none());
        assert!(config.auto_init);
    }

    #[test]
    fn test_storage_config_file() {
        let config = StorageConfig::file("/tmp/test.db");
        assert_eq!(config.db_url, "/tmp/test.db");
    }

    #[test]
    fn test_storage_config_turso() {
        let config = StorageConfig::turso("libsql://test.turso.io", "token123");
        assert_eq!(config.db_url, "libsql://test.turso.io");
        assert_eq!(config.auth_token, Some("token123".to_string()));
    }
}
