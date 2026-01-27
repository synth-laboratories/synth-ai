//! Tracing error types.

use thiserror::Error;

/// Errors that can occur during tracing operations.
#[derive(Debug, Error)]
pub enum TracingError {
    /// Database connection or query error
    #[error("database error: {0}")]
    Database(String),

    /// Session not started
    #[error("no active session")]
    NoActiveSession,

    /// Timestep not started
    #[error("no active timestep")]
    NoActiveTimestep,

    /// Session already active
    #[error("session already active: {0}")]
    SessionAlreadyActive(String),

    /// Invalid state transition
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Storage initialization error
    #[error("storage not initialized")]
    NotInitialized,

    /// Generic internal error
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for TracingError {
    fn from(err: serde_json::Error) -> Self {
        TracingError::Serialization(err.to_string())
    }
}

impl From<libsql::Error> for TracingError {
    fn from(err: libsql::Error) -> Self {
        TracingError::Database(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TracingError::NoActiveSession;
        assert_eq!(err.to_string(), "no active session");

        let err = TracingError::Database("connection failed".to_string());
        assert!(err.to_string().contains("connection failed"));
    }
}
