"""Core data structures for tracing v3."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TimeRecord:
    """Time information for events and messages."""
    event_time: float
    message_time: Optional[int] = None


@dataclass
class SessionEventMessage:
    """Message exchanged during a session."""
    content: str
    message_type: str
    time_record: TimeRecord
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseEvent:
    """Base class for all event types."""
    system_instance_id: str
    time_record: TimeRecord
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: Optional[List[Any]] = None


@dataclass
class RuntimeEvent(BaseEvent):
    """Event from runtime system."""
    actions: List[int] = field(default_factory=list)


@dataclass
class EnvironmentEvent(BaseEvent):
    """Event from environment."""
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None


@dataclass
class LMCAISEvent(BaseEvent):
    """Extended CAIS event for language model interactions."""
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
    """A logical timestep within a session."""
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
    """Complete trace of a session."""
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    session_time_steps: List[SessionTimeStep] = field(default_factory=list)
    event_history: List[BaseEvent] = field(default_factory=list)
    message_history: List[SessionEventMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)