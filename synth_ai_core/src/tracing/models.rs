//! Tracing data models.
//!
//! This module contains all the data structures for trace recording,
//! corresponding to Python's `synth_ai.data.traces` and `synth_ai.data.llm_calls`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ============================================================================
// TIME & CONTENT
// ============================================================================

/// Time record for events and messages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeRecord {
    /// Unix timestamp for the event
    pub event_time: f64,
    /// Optional message-specific timestamp
    #[serde(default)]
    pub message_time: Option<i64>,
}

impl TimeRecord {
    /// Create a new time record with the current time.
    pub fn now() -> Self {
        Self {
            event_time: Utc::now().timestamp_millis() as f64 / 1000.0,
            message_time: None,
        }
    }
}

/// Content for messages, supporting text or JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageContent {
    /// Plain text content
    #[serde(default)]
    pub text: Option<String>,
    /// JSON-serialized content
    #[serde(default)]
    pub json_payload: Option<String>,
}

impl MessageContent {
    /// Create from text.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            json_payload: None,
        }
    }

    /// Create from JSON value.
    pub fn from_json(value: &Value) -> Self {
        Self {
            text: None,
            json_payload: Some(value.to_string()),
        }
    }

    /// Get content as text (either raw text or JSON string).
    pub fn as_text(&self) -> Option<&str> {
        self.text.as_deref().or(self.json_payload.as_deref())
    }
}

// ============================================================================
// EVENTS
// ============================================================================

/// Event type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// LLM/CAIS call event
    Cais,
    /// Environment step event (Gym-style)
    Environment,
    /// Runtime/action selection event
    Runtime,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Cais => write!(f, "cais"),
            EventType::Environment => write!(f, "environment"),
            EventType::Runtime => write!(f, "runtime"),
        }
    }
}

/// Base fields common to all events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaseEventFields {
    /// System/component instance ID
    pub system_instance_id: String,
    /// Time record
    pub time_record: TimeRecord,
    /// Event metadata (key-value pairs)
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Structured event metadata
    #[serde(default)]
    pub event_metadata: Option<Vec<Value>>,
}

impl BaseEventFields {
    /// Create new base fields with a system instance ID.
    pub fn new(system_instance_id: impl Into<String>) -> Self {
        Self {
            system_instance_id: system_instance_id.into(),
            time_record: TimeRecord::now(),
            metadata: HashMap::new(),
            event_metadata: None,
        }
    }
}

/// LLM/CAIS call event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LMCAISEvent {
    /// Base event fields
    #[serde(flatten)]
    pub base: BaseEventFields,
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model_name: String,
    /// Provider (e.g., "openai", "anthropic")
    #[serde(default)]
    pub provider: Option<String>,
    /// Input/prompt tokens
    #[serde(default)]
    pub input_tokens: Option<i32>,
    /// Output/completion tokens
    #[serde(default)]
    pub output_tokens: Option<i32>,
    /// Total tokens
    #[serde(default)]
    pub total_tokens: Option<i32>,
    /// Cost in USD
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// Latency in milliseconds
    #[serde(default)]
    pub latency_ms: Option<i32>,
    /// OpenTelemetry span ID
    #[serde(default)]
    pub span_id: Option<String>,
    /// OpenTelemetry trace ID
    #[serde(default)]
    pub trace_id: Option<String>,
    /// Detailed call records
    #[serde(default)]
    pub call_records: Option<Vec<LLMCallRecord>>,
    /// System state before the call
    #[serde(default)]
    pub system_state_before: Option<Value>,
    /// System state after the call
    #[serde(default)]
    pub system_state_after: Option<Value>,
}

/// Environment step event (Gymnasium/OpenAI Gym style).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentEvent {
    /// Base event fields
    #[serde(flatten)]
    pub base: BaseEventFields,
    /// Reward signal
    #[serde(default)]
    pub reward: f64,
    /// Episode terminated flag
    #[serde(default)]
    pub terminated: bool,
    /// Episode truncated flag
    #[serde(default)]
    pub truncated: bool,
    /// System state before step
    #[serde(default)]
    pub system_state_before: Option<Value>,
    /// System state after step (observations)
    #[serde(default)]
    pub system_state_after: Option<Value>,
}

/// Runtime/action selection event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeEvent {
    /// Base event fields
    #[serde(flatten)]
    pub base: BaseEventFields,
    /// Action indices/selections
    #[serde(default)]
    pub actions: Vec<i32>,
}

/// Unified event type using tagged enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum TracingEvent {
    /// LLM/CAIS call
    Cais(LMCAISEvent),
    /// Environment step
    Environment(EnvironmentEvent),
    /// Runtime action
    Runtime(RuntimeEvent),
}

impl TracingEvent {
    /// Get the event type.
    pub fn event_type(&self) -> EventType {
        match self {
            TracingEvent::Cais(_) => EventType::Cais,
            TracingEvent::Environment(_) => EventType::Environment,
            TracingEvent::Runtime(_) => EventType::Runtime,
        }
    }

    /// Get the base event fields.
    pub fn base(&self) -> &BaseEventFields {
        match self {
            TracingEvent::Cais(e) => &e.base,
            TracingEvent::Environment(e) => &e.base,
            TracingEvent::Runtime(e) => &e.base,
        }
    }

    /// Get the time record.
    pub fn time_record(&self) -> &TimeRecord {
        &self.base().time_record
    }

    /// Get the system instance ID.
    pub fn system_instance_id(&self) -> &str {
        &self.base().system_instance_id
    }
}

// ============================================================================
// LLM CALL RECORDS
// ============================================================================

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMUsage {
    #[serde(default)]
    pub input_tokens: Option<i32>,
    #[serde(default)]
    pub output_tokens: Option<i32>,
    #[serde(default)]
    pub total_tokens: Option<i32>,
    #[serde(default)]
    pub reasoning_tokens: Option<i32>,
    #[serde(default)]
    pub reasoning_input_tokens: Option<i32>,
    #[serde(default)]
    pub reasoning_output_tokens: Option<i32>,
    #[serde(default)]
    pub cache_read_tokens: Option<i32>,
    #[serde(default)]
    pub cache_write_tokens: Option<i32>,
    #[serde(default)]
    pub billable_input_tokens: Option<i32>,
    #[serde(default)]
    pub billable_output_tokens: Option<i32>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

/// Provider request parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMRequestParams {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i32>,
    #[serde(default)]
    pub n: Option<i32>,
    #[serde(default)]
    pub best_of: Option<i32>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub json_mode: Option<bool>,
    #[serde(default)]
    pub tool_config: Option<Value>,
    #[serde(default)]
    pub raw_params: HashMap<String, Value>,
}

/// LLM message content part.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMContentPart {
    /// Content type (text, image, audio, etc.)
    #[serde(rename = "type")]
    pub content_type: String,
    /// Text content
    #[serde(default)]
    pub text: Option<String>,
    /// Generic data payload
    #[serde(default)]
    pub data: Option<Value>,
    /// MIME type for media
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub base64_data: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<i64>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub width: Option<i32>,
    #[serde(default)]
    pub height: Option<i32>,
    #[serde(default)]
    pub duration_ms: Option<i32>,
    #[serde(default)]
    pub sample_rate: Option<i32>,
    #[serde(default)]
    pub channels: Option<i32>,
    #[serde(default)]
    pub language: Option<String>,
}

impl LLMContentPart {
    /// Create a text content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: Some(text.into()),
            data: None,
            mime_type: None,
            uri: None,
            base64_data: None,
            size_bytes: None,
            sha256: None,
            width: None,
            height: None,
            duration_ms: None,
            sample_rate: None,
            channels: None,
            language: None,
        }
    }
}

/// LLM message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMMessage {
    /// Role (system, user, assistant, tool)
    pub role: String,
    /// Message content parts
    #[serde(default)]
    pub parts: Vec<LLMContentPart>,
    /// Optional message name
    #[serde(default)]
    pub name: Option<String>,
    /// Tool call ID for tool messages
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl LLMMessage {
    /// Create a simple text message.
    pub fn new(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            parts: vec![LLMContentPart::text(text)],
            name: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the text content of the message.
    pub fn text(&self) -> Option<&str> {
        self.parts.iter().find_map(|p| p.text.as_deref())
    }
}

/// Tool call specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallSpec {
    /// Tool/function name
    pub name: String,
    /// Arguments as JSON string
    pub arguments_json: String,
    /// Parsed arguments (optional)
    #[serde(default)]
    pub arguments: Option<Value>,
    /// Call ID
    #[serde(default)]
    pub call_id: Option<String>,
    /// Index in batch
    #[serde(default)]
    pub index: Option<i32>,
    /// Parent call ID
    #[serde(default)]
    pub parent_call_id: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Tool call result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Correlates to ToolCallSpec
    #[serde(default)]
    pub call_id: Option<String>,
    /// Execution result text
    #[serde(default)]
    pub output_text: Option<String>,
    /// Exit code
    #[serde(default)]
    pub exit_code: Option<i32>,
    /// Status (ok, error)
    #[serde(default)]
    pub status: Option<String>,
    /// Error message
    #[serde(default)]
    pub error_message: Option<String>,
    /// Start timestamp
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    /// Completion timestamp
    #[serde(default)]
    pub completed_at: Option<DateTime<Utc>>,
    /// Duration in milliseconds
    #[serde(default)]
    pub duration_ms: Option<i32>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Optional streaming chunk representation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMChunk {
    pub sequence_index: i32,
    pub received_at: DateTime<Utc>,
    #[serde(default)]
    pub event_type: Option<String>,
    #[serde(default)]
    pub choice_index: Option<i32>,
    #[serde(default)]
    pub raw_json: Option<String>,
    #[serde(default)]
    pub delta_text: Option<String>,
    #[serde(default)]
    pub delta: Option<Value>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Normalized LLM call record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMCallRecord {
    /// Unique call ID
    pub call_id: String,
    /// API type (chat_completions, completions, responses)
    pub api_type: String,
    /// Provider (openai, anthropic, etc.)
    #[serde(default)]
    pub provider: Option<String>,
    /// Model name
    #[serde(default)]
    pub model_name: String,
    /// Schema version
    #[serde(default)]
    pub schema_version: Option<String>,
    /// Call start time
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    /// Call completion time
    #[serde(default)]
    pub completed_at: Option<DateTime<Utc>>,
    /// Latency in milliseconds
    #[serde(default)]
    pub latency_ms: Option<i32>,
    /// Provider request parameters
    #[serde(default)]
    pub request_params: LLMRequestParams,
    /// Input messages
    #[serde(default)]
    pub input_messages: Vec<LLMMessage>,
    /// Input text (completions-style)
    #[serde(default)]
    pub input_text: Option<String>,
    /// Tool choice
    #[serde(default)]
    pub tool_choice: Option<String>,
    /// Output messages
    #[serde(default)]
    pub output_messages: Vec<LLMMessage>,
    /// Output choices (n>1)
    #[serde(default)]
    pub outputs: Vec<LLMMessage>,
    /// Output text (completions-style)
    #[serde(default)]
    pub output_text: Option<String>,
    /// Tool calls in response
    #[serde(default)]
    pub output_tool_calls: Vec<ToolCallSpec>,
    /// Tool execution results
    #[serde(default)]
    pub tool_results: Vec<ToolCallResult>,
    /// Token usage
    #[serde(default)]
    pub usage: Option<LLMUsage>,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Choice index
    #[serde(default)]
    pub choice_index: Option<i32>,
    /// Streaming chunks
    #[serde(default)]
    pub chunks: Option<Vec<LLMChunk>>,
    /// Raw request JSON
    #[serde(default)]
    pub request_raw_json: Option<String>,
    /// Raw response JSON
    #[serde(default)]
    pub response_raw_json: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Provider request ID
    #[serde(default)]
    pub provider_request_id: Option<String>,
    /// Request server timing info
    #[serde(default)]
    pub request_server_timing: Option<Value>,
    /// Outcome status
    #[serde(default)]
    pub outcome: Option<String>,
    /// Error details
    #[serde(default)]
    pub error: Option<Value>,
    /// Token trace info
    #[serde(default)]
    pub token_traces: Option<Vec<Value>>,
    /// Safety metadata
    #[serde(default)]
    pub safety: Option<Value>,
    /// Refusal metadata
    #[serde(default)]
    pub refusal: Option<Value>,
    /// Redactions
    #[serde(default)]
    pub redactions: Option<Vec<Value>>,
}

// ============================================================================
// SESSION STRUCTURE
// ============================================================================

/// Inter-system message (Markov blanket).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkovBlanketMessage {
    /// Message content
    pub content: MessageContent,
    /// Message type (user, assistant, system, tool_use, tool_result)
    pub message_type: String,
    /// Time record
    pub time_record: TimeRecord,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// A timestep within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTimeStep {
    /// Unique step ID
    pub step_id: String,
    /// Sequential step index
    pub step_index: i32,
    /// Step start time
    pub timestamp: DateTime<Utc>,
    /// Conversation turn number
    #[serde(default)]
    pub turn_number: Option<i32>,
    /// Events in this step
    #[serde(default)]
    pub events: Vec<TracingEvent>,
    /// Messages in this step
    #[serde(default)]
    pub markov_blanket_messages: Vec<MarkovBlanketMessage>,
    /// Step-specific metadata
    #[serde(default)]
    pub step_metadata: HashMap<String, Value>,
    /// Step completion time
    #[serde(default)]
    pub completed_at: Option<DateTime<Utc>>,
}

impl SessionTimeStep {
    /// Create a new timestep.
    pub fn new(step_id: impl Into<String>, step_index: i32) -> Self {
        Self {
            step_id: step_id.into(),
            step_index,
            timestamp: Utc::now(),
            turn_number: None,
            events: Vec::new(),
            markov_blanket_messages: Vec::new(),
            step_metadata: HashMap::new(),
            completed_at: None,
        }
    }

    /// Mark the timestep as complete.
    pub fn complete(&mut self) {
        self.completed_at = Some(Utc::now());
    }
}

/// A complete session trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTrace {
    /// Session ID
    pub session_id: String,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Ordered timesteps
    #[serde(default)]
    pub session_time_steps: Vec<SessionTimeStep>,
    /// Flattened event history
    #[serde(default)]
    pub event_history: Vec<TracingEvent>,
    /// Flattened message history
    #[serde(default)]
    pub markov_blanket_message_history: Vec<MarkovBlanketMessage>,
    /// Session-level metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl SessionTrace {
    /// Create a new session trace.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            created_at: Utc::now(),
            session_time_steps: Vec::new(),
            event_history: Vec::new(),
            markov_blanket_message_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the number of timesteps.
    pub fn num_timesteps(&self) -> usize {
        self.session_time_steps.len()
    }

    /// Get the number of events.
    pub fn num_events(&self) -> usize {
        self.event_history.len()
    }

    /// Get the number of messages.
    pub fn num_messages(&self) -> usize {
        self.markov_blanket_message_history.len()
    }
}

// ============================================================================
// REWARDS
// ============================================================================

/// Session-level outcome reward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeReward {
    /// Objective key (default: "reward")
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Total reward value
    pub total_reward: f64,
    /// Number of achievements
    #[serde(default)]
    pub achievements_count: i32,
    /// Total steps in session
    #[serde(default)]
    pub total_steps: i32,
    /// Additional metadata
    #[serde(default)]
    pub reward_metadata: HashMap<String, Value>,
    /// Annotation
    #[serde(default)]
    pub annotation: Option<Value>,
}

fn default_objective_key() -> String {
    "reward".to_string()
}

/// Event-level reward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventReward {
    /// Objective key (default: "reward")
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Reward value
    pub reward_value: f64,
    /// Reward type (shaped, sparse, achievement, penalty, evaluator, human)
    #[serde(default)]
    pub reward_type: Option<String>,
    /// Key (e.g., achievement name)
    #[serde(default)]
    pub key: Option<String>,
    /// Annotation
    #[serde(default)]
    pub annotation: Option<Value>,
    /// Source (environment, runner, evaluator, human)
    #[serde(default)]
    pub source: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_record() {
        let tr = TimeRecord::now();
        assert!(tr.event_time > 0.0);
        assert!(tr.message_time.is_none());
    }

    #[test]
    fn test_message_content() {
        let mc = MessageContent::from_text("hello");
        assert_eq!(mc.as_text(), Some("hello"));

        let mc = MessageContent::from_json(&serde_json::json!({"key": "value"}));
        assert!(mc.json_payload.is_some());
    }

    #[test]
    fn test_event_serialization() {
        let event = TracingEvent::Cais(LMCAISEvent {
            base: BaseEventFields::new("test-system"),
            model_name: "gpt-4".to_string(),
            provider: Some("openai".to_string()),
            input_tokens: Some(100),
            output_tokens: Some(50),
            ..Default::default()
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("cais"));
        assert!(json.contains("gpt-4"));

        let parsed: TracingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.event_type(), EventType::Cais);
    }

    #[test]
    fn test_session_trace() {
        let mut trace = SessionTrace::new("test-session");
        assert_eq!(trace.num_timesteps(), 0);

        let step = SessionTimeStep::new("step-1", 0);
        trace.session_time_steps.push(step);
        assert_eq!(trace.num_timesteps(), 1);
    }

    #[test]
    fn test_llm_message() {
        let msg = LLMMessage::new("user", "Hello, world!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.text(), Some("Hello, world!"));
    }
}
