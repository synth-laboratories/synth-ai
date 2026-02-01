//! Tracing system for recording session traces.
//!
//! This module provides a comprehensive tracing system for recording
//! LLM calls, environment interactions, and other events during
//! agent execution sessions.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    User Code                            │
//! └────────────────────────┬────────────────────────────────┘
//!                          │
//!               ┌──────────▼──────────┐
//!               │   SessionTracer     │ ◄── HookManager (callbacks)
//!               │   - start_session() │
//!               │   - start_timestep()│
//!               │   - record_event()  │
//!               │   - record_message()│
//!               └──────────┬──────────┘
//!                          │
//!               ┌──────────▼──────────┐
//!               │   TraceStorage      │ (trait)
//!               └──────────┬──────────┘
//!                          │
//!               ┌──────────▼──────────┐
//!               │ LibsqlTraceStorage  │ ◄── SQLite / Turso
//!               └─────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use synth_ai_core::tracing::{SessionTracer, LibsqlTraceStorage, TracingEvent, LMCAISEvent, BaseEventFields};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create storage (in-memory for testing)
//!     let storage = Arc::new(LibsqlTraceStorage::new_memory().await?);
//!
//!     // Create tracer
//!     let tracer = SessionTracer::new(storage);
//!
//!     // Start session
//!     let session_id = tracer.start_session(None, Default::default()).await?;
//!
//!     // Record a timestep with an LLM call
//!     tracer.start_timestep("step1", Some(1), Default::default()).await?;
//!
//!     let event = TracingEvent::Cais(LMCAISEvent {
//!         base: BaseEventFields::new("llm-agent"),
//!         model_name: "gpt-4".to_string(),
//!         provider: Some("openai".to_string()),
//!         input_tokens: Some(150),
//!         output_tokens: Some(50),
//!         cost_usd: Some(0.006),
//!         latency_ms: Some(1200),
//!         ..Default::default()
//!     });
//!
//!     tracer.record_event(event).await?;
//!     tracer.end_timestep().await?;
//!
//!     // End session
//!     let trace = tracer.end_session(true).await?;
//!     println!("Session complete: {} events", trace.event_history.len());
//!
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod hooks;
pub mod libsql_storage;
pub mod models;
pub mod storage;
pub mod tracer;
pub mod utils;

// Re-export main types for convenience
pub use error::TracingError;
pub use hooks::{HookCallback, HookContext, HookEvent, HookManager};
pub use libsql_storage::LibsqlTraceStorage;
pub use models::{
    BaseEventFields, EnvironmentEvent, EventReward, EventType, LLMCallRecord, LLMChunk,
    LLMContentPart, LLMMessage, LLMRequestParams, LLMUsage, LMCAISEvent, MarkovBlanketMessage,
    MessageContent, OutcomeReward, RuntimeEvent, SessionTimeStep, SessionTrace, TimeRecord,
    ToolCallResult, ToolCallSpec, TracingEvent,
};
pub use storage::{QueryParams, StorageConfig, TraceStorage};
pub use tracer::SessionTracer;
pub use utils::{calculate_cost, detect_provider};
