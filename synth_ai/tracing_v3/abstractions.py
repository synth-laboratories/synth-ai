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
    - Messages: Information passed between subsystems (user, agent, runtime, environments)

Concepts:
---------
- Events capture something that happened inside a subsystem. They may or may not be externally
  visible. Examples include an LLM API call (LMCAISEvent), a tool selection (RuntimeEvent), or
  a tool execution outcome (EnvironmentEvent).

- Messages represent information transmitted between subsystems within the session.
  Messages are used to record communications like: a user sending input to the agent,
  the agent/runtime sending a tool invocation to an environment, the environment sending a
  tool result back, and the agent sending a reply to the user. Do not confuse these with
  provider-specific LLM API "messages" (prompt formatting) — those belong inside an LMCAISEvent
  as part of its input/output content, not as SessionEventMessages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .lm_call_record_abstractions import LLMCallRecord


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
    message_time: int | None = None


@dataclass(frozen=True)
class SessionMessageContent:
    """Normalized payload stored alongside session messages."""

    text: str | None = None
    json_payload: str | None = None

    def as_text(self) -> str:
        return self.text or (self.json_payload or "")

    def has_json(self) -> bool:
        return self.json_payload is not None

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.as_text()


@dataclass
class SessionEventMarkovBlanketMessage:
    """Message crossing Markov blanket boundaries between systems in a session.

    IMPORTANT: This represents information transfer BETWEEN distinct systems/subsystems,
    where each system is conceptualized as having a Markov blanket that separates its
    internal states from the external environment. These messages cross those boundaries.

    This is NOT for chat messages within an LLM conversation (those belong in LLMCallRecord).
    Instead, this captures inter-system communication such as:
    - Human -> Agent system (user providing instructions)
    - Agent -> Runtime (agent deciding on an action)
    - Runtime -> Environment (executing a tool/action)
    - Environment -> Runtime (returning results)
    - Runtime -> Agent (passing back results)
    - Agent -> Human (final response)

    Each system maintains its own internal state and processing, but can only influence
    other systems through these explicit boundary-crossing messages. This follows the
    Free Energy Principle where systems minimize surprise by maintaining boundaries.

    Attributes:
        content: The actual message content crossing the boundary (text, JSON, etc.)
        message_type: Type of boundary crossing (e.g., 'observation', 'action', 'result')
        time_record: Timing information for the boundary crossing
        metadata: Boundary crossing metadata. Recommended keys:
                  - 'step_id': Timestep identifier
                  - 'from_system_instance_id': UUID of the sending system
                  - 'to_system_instance_id': UUID of the receiving system
                  - 'from_system_role': Role of sender (e.g., 'human', 'agent', 'runtime', 'environment')
                  - 'to_system_role': Role of receiver
                  - 'boundary_type': Type of Markov blanket boundary being crossed
                  - 'call_id': Correlate request/response pairs across boundaries
                  - 'causal_influence': Direction of causal flow
    """

    content: SessionMessageContent
    message_type: str
    time_record: TimeRecord
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseEvent:
    """Base class for all event types.

    This is the foundation for all events in the tracing system. Every event must
    have a system identifier and timing information. Events are intra-system facts
    (they occur within a subsystem) and are not necessarily direct communications.

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
    metadata: dict[str, Any] = field(default_factory=dict)
    event_metadata: list[Any] | None = None


@dataclass
class RuntimeEvent(BaseEvent):
    """Event from runtime system.

    Captures events from the AI system's runtime, typically representing decisions
    or actions taken by the system (e.g., selecting a tool with arguments).
    Use paired SessionEventMessages to record the communication of this choice to
    the environment.

    Attributes:
        actions: List of action identifiers or indices. The interpretation
                depends on the system (e.g., discrete action indices for RL,
                tool selection IDs for agents, etc.)
    """

    actions: list[int] = field(default_factory=list)


@dataclass
class EnvironmentEvent(BaseEvent):
    """Event from environment.

    Captures feedback from the environment in response to system actions (e.g.,
    command output, exit codes, observations). Use a paired SessionEventMessage
    to record the environment-to-agent communication of the result.
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
    system_state_before: dict[str, Any] | None = None
    system_state_after: dict[str, Any] | None = None


@dataclass
class LMCAISEvent(BaseEvent):
    """Extended CAIS event for language model interactions.

    CAIS (Claude AI System) events capture detailed information about LLM calls,
    including performance metrics, cost tracking, and distributed tracing support.
    Treat provider-specific prompt/completion structures as part of this event's
    data. Do not emit them as SessionEventMessages.

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
        call_records: List of normalized LLM call records capturing request/response
                      details (messages, tool calls/results, usage, params, etc.).
    """

    model_name: str = ""
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    span_id: str | None = None
    trace_id: str | None = None
    system_state_before: dict[str, Any] | None = None
    system_state_after: dict[str, Any] | None = None
    call_records: list[LLMCallRecord] = field(default_factory=list)


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    turn_number: int | None = None
    events: list[BaseEvent] = field(default_factory=list)
    markov_blanket_messages: list[SessionEventMarkovBlanketMessage] = field(default_factory=list)
    step_metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime | None = None


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
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    session_time_steps: list[SessionTimeStep] = field(default_factory=list)
    event_history: list[BaseEvent] = field(default_factory=list)
    markov_blanket_message_history: list[SessionEventMarkovBlanketMessage] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    session_metadata: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            A dictionary containing all session data, suitable for
            JSON serialization or database storage.
        """
        return asdict(self)
