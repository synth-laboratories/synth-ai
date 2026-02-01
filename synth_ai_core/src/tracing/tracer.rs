//! Session tracer - main API for recording traces.
//!
//! This module provides `SessionTracer`, the high-level API for recording
//! session traces with timesteps, events, and messages.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::error::TracingError;
use super::hooks::{HookContext, HookEvent, HookManager};
use super::models::{
    EventReward, MarkovBlanketMessage, MessageContent, OutcomeReward, SessionTimeStep,
    SessionTrace, TimeRecord, TracingEvent,
};
use super::storage::{QueryParams, TraceStorage};

/// Session tracer for recording traces.
///
/// Provides a high-level API for recording session traces with timesteps,
/// events, and messages. Supports real-time persistence and hook callbacks.
///
/// # Example
///
/// ```ignore
/// use synth_ai_core::tracing::{SessionTracer, LibsqlTraceStorage, TracingEvent, LMCAISEvent};
/// use std::sync::Arc;
///
/// let storage = Arc::new(LibsqlTraceStorage::new_memory().await?);
/// let tracer = SessionTracer::new(storage);
///
/// let session_id = tracer.start_session(None, Default::default()).await?;
/// tracer.start_timestep("step-1", Some(1), Default::default()).await?;
/// tracer.record_event(TracingEvent::Cais(LMCAISEvent { ... })).await?;
/// tracer.end_timestep().await?;
/// let trace = tracer.end_session(true).await?;
/// ```
pub struct SessionTracer {
    /// Storage backend
    storage: Arc<dyn TraceStorage>,
    /// Hook manager
    hooks: Mutex<HookManager>,
    /// Current session trace (if any)
    current_session: Mutex<Option<SessionTrace>>,
    /// Current timestep (if any)
    current_step: Mutex<Option<CurrentStep>>,
    /// Whether to auto-save to storage
    auto_save: bool,
}

/// Tracks the current timestep with its database ID.
struct CurrentStep {
    step: SessionTimeStep,
    db_id: Option<i64>,
}

impl SessionTracer {
    /// Create a new session tracer with a storage backend.
    pub fn new(storage: Arc<dyn TraceStorage>) -> Self {
        Self {
            storage,
            hooks: Mutex::new(HookManager::new()),
            current_session: Mutex::new(None),
            current_step: Mutex::new(None),
            auto_save: true,
        }
    }

    /// Create a new session tracer with custom hooks.
    pub fn with_hooks(storage: Arc<dyn TraceStorage>, hooks: HookManager) -> Self {
        Self {
            storage,
            hooks: Mutex::new(hooks),
            current_session: Mutex::new(None),
            current_step: Mutex::new(None),
            auto_save: true,
        }
    }

    /// Set whether to auto-save events to storage.
    pub fn set_auto_save(&mut self, auto_save: bool) {
        self.auto_save = auto_save;
    }

    /// Register a hook callback.
    pub async fn register_hook(
        &self,
        event: HookEvent,
        callback: super::hooks::HookCallback,
        priority: i32,
    ) {
        let mut hooks = self.hooks.lock().await;
        hooks.register(event, callback, priority);
    }

    // ========================================================================
    // SESSION LIFECYCLE
    // ========================================================================

    /// Start a new session.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Optional session ID. If None, a UUID will be generated.
    /// * `metadata` - Session-level metadata.
    ///
    /// # Returns
    ///
    /// The session ID.
    pub async fn start_session(
        &self,
        session_id: Option<&str>,
        metadata: HashMap<String, Value>,
    ) -> Result<String, TracingError> {
        let mut current = self.current_session.lock().await;

        if current.is_some() {
            return Err(TracingError::SessionAlreadyActive(
                current.as_ref().unwrap().session_id.clone(),
            ));
        }

        let session_id = session_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let mut trace = SessionTrace::new(&session_id);
        trace.metadata = metadata;

        // Persist to storage
        if self.auto_save {
            self.storage
                .ensure_session(
                    &session_id,
                    trace.created_at,
                    &serde_json::to_value(&trace.metadata).unwrap_or_default(),
                )
                .await?;
        }

        // Trigger hook
        let context = HookContext::new().with_session(&session_id);
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::SessionStart, &context);

        *current = Some(trace);

        Ok(session_id)
    }

    /// End the current session.
    ///
    /// # Arguments
    ///
    /// * `save` - Whether to save the session to storage (ignored if auto_save is true).
    ///
    /// # Returns
    ///
    /// The completed session trace.
    pub async fn end_session(&self, save: bool) -> Result<SessionTrace, TracingError> {
        // End any active timestep first
        if self.current_step.lock().await.is_some() {
            self.end_timestep().await?;
        }

        let mut current = self.current_session.lock().await;

        let trace = current.take().ok_or(TracingError::NoActiveSession)?;

        // Update session counts
        if self.auto_save || save {
            self.storage
                .update_session_counts(&trace.session_id)
                .await?;
        }

        // Trigger hook
        let context = HookContext::new().with_session(&trace.session_id);
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::SessionEnd, &context);

        Ok(trace)
    }

    /// Get the current session ID (if any).
    pub async fn current_session_id(&self) -> Option<String> {
        self.current_session
            .lock()
            .await
            .as_ref()
            .map(|s| s.session_id.clone())
    }

    /// Execute a raw SQL query against the underlying storage.
    pub async fn query(&self, sql: &str, params: QueryParams) -> Result<Vec<Value>, TracingError> {
        self.storage.query(sql, params).await
    }

    // ========================================================================
    // TIMESTEP LIFECYCLE
    // ========================================================================

    /// Start a new timestep.
    ///
    /// # Arguments
    ///
    /// * `step_id` - Unique step identifier.
    /// * `turn_number` - Optional conversation turn number.
    /// * `metadata` - Step-level metadata.
    pub async fn start_timestep(
        &self,
        step_id: &str,
        turn_number: Option<i32>,
        metadata: HashMap<String, Value>,
    ) -> Result<(), TracingError> {
        let (session_id, step_index) = {
            let session_guard = self.current_session.lock().await;
            let session = session_guard
                .as_ref()
                .ok_or(TracingError::NoActiveSession)?;
            (
                session.session_id.clone(),
                session.session_time_steps.len() as i32,
            )
        };

        // End any previous timestep
        if self.current_step.lock().await.is_some() {
            self.end_timestep().await?;
        }

        let mut step = SessionTimeStep::new(step_id, step_index);
        step.turn_number = turn_number;
        step.step_metadata = metadata;

        // Persist to storage
        let db_id = if self.auto_save {
            Some(self.storage.ensure_timestep(&session_id, &step).await?)
        } else {
            None
        };

        // Trigger hook
        let context = HookContext::new()
            .with_session(&session_id)
            .with_step(step_id);
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::TimestepStart, &context);

        *self.current_step.lock().await = Some(CurrentStep { step, db_id });

        Ok(())
    }

    /// End the current timestep.
    pub async fn end_timestep(&self) -> Result<(), TracingError> {
        let session_id = self
            .current_session_id()
            .await
            .ok_or(TracingError::NoActiveSession)?;

        let mut current_step = self.current_step.lock().await;
        let mut step_data = current_step.take().ok_or(TracingError::NoActiveTimestep)?;

        step_data.step.complete();

        // Update storage
        if self.auto_save {
            self.storage
                .update_timestep(
                    &session_id,
                    &step_data.step.step_id,
                    step_data.step.completed_at,
                )
                .await?;
        }

        // Trigger hook
        let context = HookContext::new()
            .with_session(&session_id)
            .with_step(&step_data.step.step_id);
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::TimestepEnd, &context);

        // Add to session
        let mut session = self.current_session.lock().await;
        if let Some(ref mut s) = *session {
            s.session_time_steps.push(step_data.step);
        }

        Ok(())
    }

    /// Get the current step ID (if any).
    pub async fn current_step_id(&self) -> Option<String> {
        self.current_step
            .lock()
            .await
            .as_ref()
            .map(|s| s.step.step_id.clone())
    }

    // ========================================================================
    // EVENT RECORDING
    // ========================================================================

    /// Record an event.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to record.
    ///
    /// # Returns
    ///
    /// The database ID of the event (if auto_save is enabled).
    pub async fn record_event(&self, event: TracingEvent) -> Result<Option<i64>, TracingError> {
        let session_id = self
            .current_session_id()
            .await
            .ok_or(TracingError::NoActiveSession)?;

        let timestep_db_id = self
            .current_step
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.db_id);

        // Persist to storage
        let event_id = if self.auto_save {
            Some(
                self.storage
                    .insert_event(&session_id, timestep_db_id, &event)
                    .await?,
            )
        } else {
            None
        };

        // Trigger hook
        let context = HookContext::new()
            .with_session(&session_id)
            .with_event(event.clone());
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::EventRecorded, &context);

        // Add to session history
        let mut session = self.current_session.lock().await;
        if let Some(ref mut s) = *session {
            s.event_history.push(event.clone());
        }

        // Add to current timestep
        let mut step = self.current_step.lock().await;
        if let Some(ref mut s) = *step {
            s.step.events.push(event);
        }

        Ok(event_id)
    }

    // ========================================================================
    // MESSAGE RECORDING
    // ========================================================================

    /// Record a message.
    ///
    /// # Arguments
    ///
    /// * `content` - Message content.
    /// * `message_type` - Type of message (user, assistant, system, tool_use, tool_result).
    /// * `metadata` - Message metadata.
    ///
    /// # Returns
    ///
    /// The database ID of the message (if auto_save is enabled).
    pub async fn record_message(
        &self,
        content: MessageContent,
        message_type: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<Option<i64>, TracingError> {
        let session_id = self
            .current_session_id()
            .await
            .ok_or(TracingError::NoActiveSession)?;

        let timestep_db_id = self
            .current_step
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.db_id);

        let msg = MarkovBlanketMessage {
            content,
            message_type: message_type.to_string(),
            time_record: TimeRecord::now(),
            metadata,
        };

        // Persist to storage
        let msg_id = if self.auto_save {
            Some(
                self.storage
                    .insert_message(&session_id, timestep_db_id, &msg)
                    .await?,
            )
        } else {
            None
        };

        // Trigger hook
        let context = HookContext::new()
            .with_session(&session_id)
            .with_message(msg.clone());
        self.hooks
            .lock()
            .await
            .trigger(HookEvent::MessageRecorded, &context);

        // Add to session history
        let mut session = self.current_session.lock().await;
        if let Some(ref mut s) = *session {
            s.markov_blanket_message_history.push(msg.clone());
        }

        // Add to current timestep
        let mut step = self.current_step.lock().await;
        if let Some(ref mut s) = *step {
            s.step.markov_blanket_messages.push(msg);
        }

        Ok(msg_id)
    }

    // ========================================================================
    // REWARD RECORDING
    // ========================================================================

    /// Record an outcome (session-level) reward.
    pub async fn record_outcome_reward(
        &self,
        reward: OutcomeReward,
    ) -> Result<Option<i64>, TracingError> {
        let session_id = self
            .current_session_id()
            .await
            .ok_or(TracingError::NoActiveSession)?;

        let reward_id = if self.auto_save {
            Some(
                self.storage
                    .insert_outcome_reward(&session_id, &reward)
                    .await?,
            )
        } else {
            None
        };

        Ok(reward_id)
    }

    /// Record an event-level reward.
    pub async fn record_event_reward(
        &self,
        event_id: i64,
        reward: EventReward,
    ) -> Result<Option<i64>, TracingError> {
        let session_id = self
            .current_session_id()
            .await
            .ok_or(TracingError::NoActiveSession)?;

        let turn_number = self
            .current_step
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.step.turn_number);

        let reward_id = if self.auto_save {
            Some(
                self.storage
                    .insert_event_reward(&session_id, event_id, None, turn_number, &reward)
                    .await?,
            )
        } else {
            None
        };

        Ok(reward_id)
    }

    // ========================================================================
    // RETRIEVAL
    // ========================================================================

    /// Get a session trace by ID from storage.
    pub async fn get_session(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionTrace>, TracingError> {
        self.storage.get_session(session_id).await
    }

    /// Delete a session from storage.
    pub async fn delete_session(&self, session_id: &str) -> Result<bool, TracingError> {
        self.storage.delete_session(session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::libsql_storage::LibsqlTraceStorage;
    use crate::tracing::models::{BaseEventFields, LMCAISEvent};

    async fn create_test_tracer() -> SessionTracer {
        let storage = Arc::new(LibsqlTraceStorage::new_memory().await.unwrap());
        SessionTracer::new(storage)
    }

    #[tokio::test]
    async fn test_session_lifecycle() {
        let tracer = create_test_tracer().await;

        // Start session
        let session_id = tracer
            .start_session(None, Default::default())
            .await
            .unwrap();
        assert!(!session_id.is_empty());
        assert_eq!(tracer.current_session_id().await, Some(session_id.clone()));

        // End session
        let trace = tracer.end_session(true).await.unwrap();
        assert_eq!(trace.session_id, session_id);
        assert!(tracer.current_session_id().await.is_none());
    }

    #[tokio::test]
    async fn test_timestep_lifecycle() {
        let tracer = create_test_tracer().await;

        tracer
            .start_session(None, Default::default())
            .await
            .unwrap();

        // Start timestep
        tracer
            .start_timestep("step-1", Some(1), Default::default())
            .await
            .unwrap();
        assert_eq!(tracer.current_step_id().await, Some("step-1".to_string()));

        // End timestep
        tracer.end_timestep().await.unwrap();
        assert!(tracer.current_step_id().await.is_none());

        let trace = tracer.end_session(true).await.unwrap();
        assert_eq!(trace.session_time_steps.len(), 1);
    }

    #[tokio::test]
    async fn test_event_recording() {
        let tracer = create_test_tracer().await;

        tracer
            .start_session(None, Default::default())
            .await
            .unwrap();
        tracer
            .start_timestep("step-1", Some(1), Default::default())
            .await
            .unwrap();

        // Record event
        let event = TracingEvent::Cais(LMCAISEvent {
            base: BaseEventFields::new("test-system"),
            model_name: "gpt-4".to_string(),
            provider: Some("openai".to_string()),
            input_tokens: Some(100),
            output_tokens: Some(50),
            ..Default::default()
        });

        let event_id = tracer.record_event(event).await.unwrap();
        assert!(event_id.is_some());

        tracer.end_timestep().await.unwrap();
        let trace = tracer.end_session(true).await.unwrap();

        assert_eq!(trace.event_history.len(), 1);
        assert_eq!(trace.session_time_steps[0].events.len(), 1);
    }

    #[tokio::test]
    async fn test_message_recording() {
        let tracer = create_test_tracer().await;

        tracer
            .start_session(None, Default::default())
            .await
            .unwrap();
        tracer
            .start_timestep("step-1", Some(1), Default::default())
            .await
            .unwrap();

        // Record message
        let content = MessageContent::from_text("Hello, world!");
        let msg_id = tracer
            .record_message(content, "user", Default::default())
            .await
            .unwrap();
        assert!(msg_id.is_some());

        tracer.end_timestep().await.unwrap();
        let trace = tracer.end_session(true).await.unwrap();

        assert_eq!(trace.markov_blanket_message_history.len(), 1);
    }

    #[tokio::test]
    async fn test_custom_session_id() {
        let tracer = create_test_tracer().await;

        let session_id = tracer
            .start_session(Some("my-custom-id"), Default::default())
            .await
            .unwrap();

        assert_eq!(session_id, "my-custom-id");
    }

    #[tokio::test]
    async fn test_session_retrieval() {
        let tracer = create_test_tracer().await;

        let session_id = tracer
            .start_session(None, Default::default())
            .await
            .unwrap();
        tracer.end_session(true).await.unwrap();

        // Retrieve from storage
        let retrieved = tracer.get_session(&session_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().session_id, session_id);
    }

    #[tokio::test]
    async fn test_no_duplicate_sessions() {
        let tracer = create_test_tracer().await;

        tracer
            .start_session(None, Default::default())
            .await
            .unwrap();

        // Try to start another session
        let result = tracer.start_session(None, Default::default()).await;
        assert!(matches!(result, Err(TracingError::SessionAlreadyActive(_))));
    }
}
