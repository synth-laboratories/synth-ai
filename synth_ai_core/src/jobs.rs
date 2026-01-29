//! Job lifecycle management.
//!
//! This module provides job status enums and a state machine for tracking
//! job lifecycle events. It mirrors the Python `synth_ai.sdk.optimization.job`
//! module for portable use across SDKs.

use crate::errors::CoreError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

/// Job lifecycle status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    /// Check if this is a terminal (final) status.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled
        )
    }

    /// Check if this is a success status.
    pub fn is_success(&self) -> bool {
        *self == JobStatus::Succeeded
    }

    /// Parse a status string (case-insensitive, handles aliases).
    ///
    /// Handles common aliases:
    /// - "success", "completed", "complete" → Succeeded
    /// - "in_progress", "running" → Running
    /// - "queued", "pending" → Pending (or Queued if exact)
    /// - "cancelled", "canceled" → Cancelled
    /// - "failed", "failure", "error" → Failed
    pub fn from_str(s: &str) -> Option<Self> {
        let normalized = s.trim().to_lowercase().replace(' ', "_");
        match normalized.as_str() {
            "pending" => Some(JobStatus::Pending),
            "queued" => Some(JobStatus::Queued),
            "running" | "in_progress" => Some(JobStatus::Running),
            "succeeded" | "success" | "completed" | "complete" => Some(JobStatus::Succeeded),
            "failed" | "failure" | "error" => Some(JobStatus::Failed),
            "cancelled" | "canceled" | "cancel" => Some(JobStatus::Cancelled),
            _ => None,
        }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Pending => "pending",
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Succeeded => "succeeded",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for JobStatus {
    fn default() -> Self {
        JobStatus::Pending
    }
}

/// Candidate status for optimization jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Evaluating,
    Completed,
    Accepted,
    Rejected,
    Failed,
}

impl CandidateStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            CandidateStatus::Completed
                | CandidateStatus::Accepted
                | CandidateStatus::Rejected
                | CandidateStatus::Failed
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CandidateStatus::Evaluating => "evaluating",
            CandidateStatus::Completed => "completed",
            CandidateStatus::Accepted => "accepted",
            CandidateStatus::Rejected => "rejected",
            CandidateStatus::Failed => "failed",
        }
    }
}

impl std::fmt::Display for CandidateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Job event types following OpenResponses schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobEventType {
    #[serde(rename = "job.created")]
    JobCreated,
    #[serde(rename = "job.queued")]
    JobQueued,
    #[serde(rename = "job.in_progress")]
    JobInProgress,
    #[serde(rename = "job.completed")]
    JobCompleted,
    #[serde(rename = "job.failed")]
    JobFailed,
    #[serde(rename = "job.cancelled")]
    JobCancelled,
}

impl JobEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobEventType::JobCreated => "job.created",
            JobEventType::JobQueued => "job.queued",
            JobEventType::JobInProgress => "job.in_progress",
            JobEventType::JobCompleted => "job.completed",
            JobEventType::JobFailed => "job.failed",
            JobEventType::JobCancelled => "job.cancelled",
        }
    }
}

/// A job lifecycle event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEvent {
    /// Event type (e.g., "job.in_progress")
    #[serde(rename = "type")]
    pub event_type: String,
    /// Job ID this event belongs to
    pub job_id: String,
    /// Sequence number (1-indexed)
    pub seq: i64,
    /// Unix timestamp (seconds since epoch)
    pub timestamp: f64,
    /// Optional event data payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Optional human-readable message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Job lifecycle state machine.
///
/// Tracks job status transitions and emits canonical events.
/// Thread-safe for single-owner use; clone for multi-threaded scenarios.
#[derive(Debug, Clone)]
pub struct JobLifecycle {
    job_id: String,
    status: JobStatus,
    events: Vec<JobEvent>,
    started_at: Option<f64>,
    ended_at: Option<f64>,
}

impl JobLifecycle {
    /// Create a new job lifecycle tracker.
    pub fn new(job_id: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            status: JobStatus::Pending,
            events: Vec::new(),
            started_at: None,
            ended_at: None,
        }
    }

    /// Get the current job status.
    pub fn status(&self) -> JobStatus {
        self.status
    }

    /// Get the job ID.
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Get elapsed time in seconds, if the job has started.
    pub fn elapsed_seconds(&self) -> Option<f64> {
        let start = self.started_at?;
        let end = self.ended_at.unwrap_or_else(now_timestamp);
        Some(end - start)
    }

    /// Get the history of events.
    pub fn events(&self) -> &[JobEvent] {
        &self.events
    }

    /// Emit an event and record it in history.
    fn emit(
        &mut self,
        event_type: JobEventType,
        data: Option<Value>,
        message: Option<&str>,
    ) -> JobEvent {
        let event = JobEvent {
            event_type: event_type.as_str().to_string(),
            job_id: self.job_id.clone(),
            seq: (self.events.len() + 1) as i64,
            timestamp: now_timestamp(),
            data,
            message: message.map(String::from),
        };
        self.events.push(event.clone());
        event
    }

    /// Start the job (transition from Pending to Running).
    ///
    /// Returns the job.in_progress event.
    pub fn start(&mut self) -> Result<JobEvent, CoreError> {
        self.start_with_data(None, None)
    }

    /// Start the job with optional data and message.
    pub fn start_with_data(
        &mut self,
        data: Option<Value>,
        message: Option<&str>,
    ) -> Result<JobEvent, CoreError> {
        if self.status != JobStatus::Pending {
            return Err(CoreError::Job(crate::errors::JobErrorInfo {
                job_id: self.job_id.clone(),
                message: format!("Cannot start job in {} status", self.status),
                code: Some("INVALID_TRANSITION".to_string()),
            }));
        }

        self.status = JobStatus::Running;
        self.started_at = Some(now_timestamp());

        Ok(self.emit(
            JobEventType::JobInProgress,
            data,
            Some(message.unwrap_or("Job started")),
        ))
    }

    /// Complete the job successfully (transition from Running to Succeeded).
    ///
    /// Returns the job.completed event.
    pub fn complete(&mut self, data: Option<Value>) -> Result<JobEvent, CoreError> {
        self.complete_with_message(data, None)
    }

    /// Complete the job with optional message.
    pub fn complete_with_message(
        &mut self,
        data: Option<Value>,
        message: Option<&str>,
    ) -> Result<JobEvent, CoreError> {
        if self.status != JobStatus::Running {
            return Err(CoreError::Job(crate::errors::JobErrorInfo {
                job_id: self.job_id.clone(),
                message: format!("Cannot complete job in {} status", self.status),
                code: Some("INVALID_TRANSITION".to_string()),
            }));
        }

        self.status = JobStatus::Succeeded;
        self.ended_at = Some(now_timestamp());

        // Add elapsed time to data
        let mut event_data = data.unwrap_or_else(|| Value::Object(Default::default()));
        if let Value::Object(ref mut map) = event_data {
            if let Some(elapsed) = self.elapsed_seconds() {
                map.insert("elapsed_seconds".to_string(), Value::from(elapsed));
            }
        }

        Ok(self.emit(
            JobEventType::JobCompleted,
            Some(event_data),
            Some(message.unwrap_or("Job completed successfully")),
        ))
    }

    /// Fail the job (transition from Running to Failed).
    ///
    /// Returns the job.failed event.
    pub fn fail(&mut self, error: Option<&str>) -> Result<JobEvent, CoreError> {
        self.fail_with_data(error, None)
    }

    /// Fail the job with optional data.
    pub fn fail_with_data(
        &mut self,
        error: Option<&str>,
        data: Option<Value>,
    ) -> Result<JobEvent, CoreError> {
        if self.status != JobStatus::Running && self.status != JobStatus::Pending {
            return Err(CoreError::Job(crate::errors::JobErrorInfo {
                job_id: self.job_id.clone(),
                message: format!("Cannot fail job in {} status", self.status),
                code: Some("INVALID_TRANSITION".to_string()),
            }));
        }

        self.status = JobStatus::Failed;
        self.ended_at = Some(now_timestamp());

        let mut event_data = data.unwrap_or_else(|| Value::Object(Default::default()));
        if let Value::Object(ref mut map) = event_data {
            if let Some(err) = error {
                map.insert("error".to_string(), Value::String(err.to_string()));
            }
            if let Some(elapsed) = self.elapsed_seconds() {
                map.insert("elapsed_seconds".to_string(), Value::from(elapsed));
            }
        }

        Ok(self.emit(
            JobEventType::JobFailed,
            Some(event_data),
            Some(error.unwrap_or("Job failed")),
        ))
    }

    /// Cancel the job (transition to Cancelled from any non-terminal state).
    ///
    /// Returns the job.cancelled event.
    pub fn cancel(&mut self) -> Result<JobEvent, CoreError> {
        self.cancel_with_message(None)
    }

    /// Cancel the job with optional message.
    pub fn cancel_with_message(&mut self, message: Option<&str>) -> Result<JobEvent, CoreError> {
        if self.status.is_terminal() {
            return Err(CoreError::Job(crate::errors::JobErrorInfo {
                job_id: self.job_id.clone(),
                message: format!("Cannot cancel job in {} status", self.status),
                code: Some("INVALID_TRANSITION".to_string()),
            }));
        }

        self.status = JobStatus::Cancelled;
        self.ended_at = Some(now_timestamp());

        let mut event_data = Value::Object(Default::default());
        if let Value::Object(ref mut map) = event_data {
            if let Some(elapsed) = self.elapsed_seconds() {
                map.insert("elapsed_seconds".to_string(), Value::from(elapsed));
            }
        }

        Ok(self.emit(
            JobEventType::JobCancelled,
            Some(event_data),
            Some(message.unwrap_or("Job cancelled")),
        ))
    }
}

/// Get current Unix timestamp in seconds.
fn now_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_from_str() {
        assert_eq!(JobStatus::from_str("pending"), Some(JobStatus::Pending));
        assert_eq!(JobStatus::from_str("RUNNING"), Some(JobStatus::Running));
        assert_eq!(JobStatus::from_str("in_progress"), Some(JobStatus::Running));
        assert_eq!(JobStatus::from_str("success"), Some(JobStatus::Succeeded));
        assert_eq!(JobStatus::from_str("completed"), Some(JobStatus::Succeeded));
        assert_eq!(JobStatus::from_str("failed"), Some(JobStatus::Failed));
        assert_eq!(JobStatus::from_str("cancelled"), Some(JobStatus::Cancelled));
        assert_eq!(JobStatus::from_str("canceled"), Some(JobStatus::Cancelled));
        assert_eq!(JobStatus::from_str("unknown"), None);
    }

    #[test]
    fn test_job_status_is_terminal() {
        assert!(!JobStatus::Pending.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(JobStatus::Succeeded.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_job_lifecycle_happy_path() {
        let mut lifecycle = JobLifecycle::new("test-job-123");
        assert_eq!(lifecycle.status(), JobStatus::Pending);

        let start_event = lifecycle.start().unwrap();
        assert_eq!(lifecycle.status(), JobStatus::Running);
        assert_eq!(start_event.event_type, "job.in_progress");
        assert_eq!(start_event.seq, 1);

        let complete_event = lifecycle.complete(None).unwrap();
        assert_eq!(lifecycle.status(), JobStatus::Succeeded);
        assert_eq!(complete_event.event_type, "job.completed");
        assert_eq!(complete_event.seq, 2);

        assert!(lifecycle.elapsed_seconds().is_some());
    }

    #[test]
    fn test_job_lifecycle_fail() {
        let mut lifecycle = JobLifecycle::new("test-job-456");
        lifecycle.start().unwrap();

        let fail_event = lifecycle.fail(Some("Something went wrong")).unwrap();
        assert_eq!(lifecycle.status(), JobStatus::Failed);
        assert_eq!(fail_event.event_type, "job.failed");

        if let Some(data) = &fail_event.data {
            assert!(data.get("error").is_some());
        }
    }

    #[test]
    fn test_job_lifecycle_cancel() {
        let mut lifecycle = JobLifecycle::new("test-job-789");
        lifecycle.start().unwrap();

        let cancel_event = lifecycle.cancel().unwrap();
        assert_eq!(lifecycle.status(), JobStatus::Cancelled);
        assert_eq!(cancel_event.event_type, "job.cancelled");
    }

    #[test]
    fn test_invalid_transitions() {
        let mut lifecycle = JobLifecycle::new("test-job-invalid");

        // Can't complete before starting
        assert!(lifecycle.complete(None).is_err());

        lifecycle.start().unwrap();

        // Can't start twice
        assert!(lifecycle.start().is_err());

        lifecycle.complete(None).unwrap();

        // Can't fail after completing
        assert!(lifecycle.fail(None).is_err());

        // Can't cancel after completing
        assert!(lifecycle.cancel().is_err());
    }
}
