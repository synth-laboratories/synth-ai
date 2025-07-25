from typing import Any, List, Type, Optional, Dict
from uuid import UUID
from dataclasses import dataclass, field

from synth_ai.core.system import System

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

@dataclass
class Session:
    systems: List[System] = field(default_factory=list)


# Session-level Metadata
@dataclass
class SessionMetadum: 
    pass


@dataclass
class SessionRewardSignal(SessionMetadum):
    pass


@dataclass
class Hypothetical: # MCTS-esque stuff
    id: Any = None
    parent_id: Any = None


# A message is Information Entering a Markov Blanket OR Leaving It
@dataclass
class SessionMessage:  # I/O
    time_record: TimeRecord = field(default_factory=TimeRecord)
    origin_system_id: Optional[UUID] = None


# An event is a system's state changing
# Runtime may be stateful
@dataclass
class SessionEvent:
    pass


@dataclass
class SystemEvent(SessionEvent):
    time_record: TimeRecord = field(default_factory=TimeRecord)
    system_instance_id: Optional[Any] = None
    system_state_before: Optional[Any] = None
    system_state_after: Optional[Any] = None


@dataclass
class EventMetadata:
    pass


@dataclass
class CAISEvent(SystemEvent):
    llm_call_records: List[Any] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    event_metadata: List[EventMetadata] = field(default_factory=list)


@dataclass
class EnvironmentEvent(SystemEvent):
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    event_metadata: List[EventMetadata] = field(default_factory=list)


@dataclass
class RuntimeEvent(SessionEvent):
    system_state_before: Optional[Any] = None
    system_state_after: Optional[Any] = None
    actions: Optional[List[Any]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


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


