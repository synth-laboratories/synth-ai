"""
Serializable versions of tracing abstractions with to_dict methods.
This is a temporary patch to make the dataclasses JSON-serializable.
"""

from typing import Any, List, Type, Optional, Dict
from uuid import UUID
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime

from synth_ai.core.system import System
from synth_ai.tracing_v2.utils import make_serializable


# Agent State
# Environment State
# Agent: Context (inbound message), Tool Call (outbound message)
# Environment: Action (inbound message), Env State (outbound message)
# Runtime Events: Tool calls -> Actions (K messages), Observations -> Context (1 message)

# Events + Message Queue
# Event is messages going into and out of a Markov blanket?
# In general, you actually can't make that assumption
# One message queue
# One notion of messages 
# One notion of event
# Global Time, Local Time
# System State, which represents a notion of "change"
# Crafter has
# Environment -> noop -> observation (env state change, message sent to queue, caught by runtime OR unknown)
# Runtime -> observation to input, message sent to queue
# Agent -> llm call -> state/message history hook (agent state change, message sent to queue, caught by runtime OR unknown)
# Runtime -> tool call(s) to actions, messages sent to queue
# Environment -> move_right -> observation (env state change, NO message sent to queue)
# Environment -> move_right -> observation (env state change, message sent to queue, caught by runtime OR unknown)

# Global time indicates messages (crossing markov blanket), local time indicates state changes / events (within markov blanket)

# Counterfactual:
# rerun llm call, flush from message step 1 onwards

@dataclass
class TimeRecord:
    event_time: Optional[Any] = None
    message_time: Optional[Any] = None
    
    def to_dict(self):
        return make_serializable(asdict(self))

@dataclass
class Session:
    systems: List[System] = field(default_factory=list)
    
    def to_dict(self):
        return make_serializable(asdict(self))


# Session-level Metadata
@dataclass
class SessionMetadum: 
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class SessionRewardSignal(SessionMetadum):
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class Hypothetical: # MCTS-esque stuff
    id: Any = None
    parent_id: Any = None
    
    def to_dict(self):
        return make_serializable(asdict(self))


# A message is Information Entering a Markov Blanket OR Leaving It
@dataclass
class SessionMessage:  # I/O
    time_record: TimeRecord = field(default_factory=TimeRecord)
    origin_system_id: Optional[UUID] = None
    
    def to_dict(self):
        return make_serializable(asdict(self))


# An event is a system's state changing
# Runtime may be stateful
@dataclass
class SessionEvent:
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class SystemEvent(SessionEvent):
    time_record: TimeRecord = field(default_factory=TimeRecord)
    system_instance_id: Optional[Any] = None
    system_state_before: Optional[Any] = None
    system_state_after: Optional[Any] = None
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class EventMetadata:
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class CAISEvent(SystemEvent):
    llm_call_records: List[Any] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    event_metadata: List[EventMetadata] = field(default_factory=list)
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class EnvironmentEvent(SystemEvent):
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    event_metadata: List[EventMetadata] = field(default_factory=list)
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class RuntimeEvent(SystemEvent):
    event_metadata: List[EventMetadata] = field(default_factory=list)
    
    def to_dict(self):
        return make_serializable(asdict(self))


# Message Types
@dataclass
class MessageInputs:
    messages: List[Any] = field(default_factory=list)
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class ToolCallMessage(SessionMessage):
    pass
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass 
class OutputMessage(SessionMessage):
    pass
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class ActionMessage(SessionMessage):
    pass
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class EnvironmentStateMessage(SessionMessage):
    pass
    
    def to_dict(self):
        return make_serializable(asdict(self))


@dataclass
class EventRewardSignal(EventMetadata):
    pass


# A session is essentially this:
# a long list of messages and events

@dataclass
class SessionTrace:
    message_history: List[SessionMessage] = field(default_factory=list)
    event_history: List[SessionEvent] = field(default_factory=list)
    session_metadata: List[SessionMetadum] = field(default_factory=list)