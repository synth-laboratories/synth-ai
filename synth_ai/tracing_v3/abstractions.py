"""Core data structures for tracing v3.

This module defines the fundamental data structures used throughout the tracing system.
All structures are implemented as frozen dataclasses for immutability and type safety.

The hierarchy is designed to support different types of events while maintaining
a consistent interface for storage and processing.

Event Type Hierarchy:
-------------------
- BaseEvent: Common fields for all events
  - RuntimeEvent: Events from the runtime system (e.g., actions taken)
  - EnvironmentEvent: Events from the environment (e.g., rewards, termination)
  - LMCAISEvent: Language model events with token/cost tracking

Session Structure:
-----------------
- SessionTrace: Top-level container for a complete session
  - SessionTimeStep: Logical steps within a session (e.g., conversation turns)
    - Events: Individual events that occurred during the timestep
    - Messages: User/assistant messages exchanged
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TimeRecord:
    """Time information for events and messages.
    
    This class captures timing information with microsecond precision for event
    correlation and performance analysis.
    
    Attributes:
        event_time: Unix timestamp (float) when the event occurred. This is the
                   primary timestamp used for ordering and correlation.
        message_time: Optional integer timestamp for message-specific timing.
                     Can be used for external message IDs or sequence numbers.
    """

    event_time: float
    message_time: Optional[int] = None


@dataclass
class SessionEventMessage:
    """Message exchanged during a session.
    
    Represents any message passed between participants in a session, including
    user inputs, assistant responses, and system messages.
    
    Attributes:
        content: The actual message content (text, JSON, etc.)
        message_type: Type identifier (e.g., 'user', 'assistant', 'system', 'tool')
        time_record: Timing information for the message
        metadata: Additional message metadata (e.g., model used, tokens consumed,
                 tool calls, attachments, etc.)
    """

    content: str
    message_type: str
    time_record: TimeRecord
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseEvent:
    """Base class for all event types.
    
    This is the foundation for all events in the tracing system. Every event
    must have a system identifier and timing information.
    
    Attributes:
        system_instance_id: Identifier for the system/component that generated
                           this event (e.g., 'llm', 'environment', 'tool_executor')
        time_record: Timing information for the event
        metadata: Flexible dictionary for event-specific data. Common keys include:
                 - 'step_id': Associated timestep identifier
                 - 'error': Error information if event failed
                 - 'duration_ms': Event duration in milliseconds
        event_metadata: Optional list for structured metadata that doesn't fit
                       in the dictionary format (e.g., tensor data, embeddings)
    """

    system_instance_id: str
    time_record: TimeRecord
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: Optional[List[Any]] = None


@dataclass
class RuntimeEvent(BaseEvent):
    """Event from runtime system.
    
    Captures events from the AI system's runtime, typically representing
    decisions or actions taken by the system.
    
    Attributes:
        actions: List of action identifiers or indices. The interpretation
                depends on the system (e.g., discrete action indices for RL,
                tool selection IDs for agents, etc.)
    """

    actions: List[int] = field(default_factory=list)


@dataclass
class EnvironmentEvent(BaseEvent):
    """Event from environment.
    
    Captures feedback from the environment in response to system actions.
    Follows the Gymnasium/OpenAI Gym convention for compatibility.
    
    Attributes:
        reward: Scalar reward signal from the environment
        terminated: Whether the episode ended due to reaching a terminal state
        truncated: Whether the episode ended due to a time/step limit
        system_state_before: System state before the action (for debugging)
        system_state_after: System state after the action (observations)
    """

    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None


@dataclass
class LMCAISEvent(BaseEvent):
    """Extended CAIS event for language model interactions.
    
    CAIS (Claude AI System) events capture detailed information about LLM calls,
    including performance metrics, cost tracking, and distributed tracing support.
    
    Attributes:
        model_name: The specific model used (e.g., 'gpt-4', 'claude-3-opus')
        provider: LLM provider (e.g., 'openai', 'anthropic', 'local')
        input_tokens: Number of tokens in the prompt/input
        output_tokens: Number of tokens in the completion/output
        total_tokens: Total tokens used (input + output)
        cost_usd: Estimated cost in US dollars for this call
        latency_ms: End-to-end latency in milliseconds
        span_id: OpenTelemetry compatible span identifier
        trace_id: OpenTelemetry compatible trace identifier
        system_state_before: State snapshot before the LLM call
        system_state_after: State snapshot after the LLM call
    """

    model_name: str = ""
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None


@dataclass
class SessionTimeStep:
    """A logical timestep within a session.
    
    Represents a discrete step in the session timeline. In conversational AI,
    this often corresponds to a single turn of dialogue. In RL systems, it
    might represent a single environment step.
    
    Attributes:
        step_id: Unique identifier for this step (e.g., 'turn_1', 'step_42')
        step_index: Sequential index of this step within the session
        timestamp: When this timestep started (UTC)
        turn_number: Optional turn number for conversational contexts
        events: All events that occurred during this timestep
        step_messages: Messages exchanged during this timestep
        step_metadata: Additional metadata specific to this step (e.g.,
                      'user_feedback', 'context_switches', 'tool_calls')
        completed_at: When this timestep was completed (None if still active)
    """

    step_id: str = ""
    step_index: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    turn_number: Optional[int] = None
    events: List[BaseEvent] = field(default_factory=list)
    step_messages: List[SessionEventMessage] = field(default_factory=list)
    step_metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[datetime] = None


@dataclass
class SessionTrace:
    """Complete trace of a session.
    
    The top-level container that holds all data for a single execution session.
    This could represent a complete conversation, an RL episode, or any other
    bounded interaction sequence.
    
    Attributes:
        session_id: Unique identifier for this session
        created_at: When the session started (UTC)
        session_time_steps: Ordered list of timesteps in this session
        event_history: Complete chronological list of all events
        message_history: Complete chronological list of all messages
        metadata: Session-level metadata (e.g., 'user_id', 'experiment_id',
                 'model_config', 'environment_name')
        session_metadata: Optional list of structured metadata entries that
                         don't fit the dictionary format
    
    Note:
        Both event_history and message_history contain the complete record,
        while individual timesteps also reference their specific events/messages.
        This redundancy enables both efficient queries and complete reconstruction.
    """

    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    session_time_steps: List[SessionTimeStep] = field(default_factory=list)
    event_history: List[BaseEvent] = field(default_factory=list)
    message_history: List[SessionEventMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            A dictionary containing all session data, suitable for
            JSON serialization or database storage.
        """
        return asdict(self)
