"""Type definitions for tracing storage."""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class EventType(str, Enum):
    """Enumeration of event types."""
    CAIS = "cais"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


class MessageType(str, Enum):
    """Enumeration of message types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class Provider(str, Enum):
    """Enumeration of LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"
    UNKNOWN = "unknown"


class SessionRecord(BaseModel):
    """Pydantic model for session trace records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str
    created_at: datetime
    num_timesteps: int = 0
    num_events: int = 0
    num_messages: int = 0
    metadata: Optional[List[Dict[str, Any]]] = None
    experiment_id: Optional[str] = None


class TimestepRecord(BaseModel):
    """Pydantic model for timestep records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[int] = None
    session_id: str
    step_id: str
    step_index: int
    timestamp: datetime
    num_events: int = 0
    num_messages: int = 0
    step_metadata: Optional[Dict[str, Any]] = None


class EventRecord(BaseModel):
    """Pydantic model for event records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[int] = None
    session_id: str
    timestep_id: Optional[int] = None
    event_type: EventType
    system_instance_id: str
    event_time: datetime
    message_time: Optional[int] = None
    
    # CAIS-specific fields
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[Provider] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Environment-specific fields
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    
    # Common fields
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    event_metadata: Optional[List[Dict[str, Any]]] = None


class MessageRecord(BaseModel):
    """Pydantic model for message records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[int] = None
    session_id: str
    timestep_id: Optional[int] = None
    message_type: MessageType
    content: Any
    timestamp: datetime
    event_time: Optional[datetime] = None
    message_time: Optional[int] = None


class ExperimentRecord(BaseModel):
    """Pydantic model for experiment records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime
    system_versions: Optional[List[Dict[str, str]]] = None


class SystemRecord(BaseModel):
    """Pydantic model for system records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    name: str
    description: str = ""


class SystemVersionRecord(BaseModel):
    """Pydantic model for system version records."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    system_id: str
    branch: str
    commit: str
    created_at: datetime
    description: str = ""