"""
SessionTracer for capturing LLM calls from langfuse OTEL and converting to CAIS events.

This module provides utilities to:
1. Capture langfuse OTEL spans during LLM calls
2. Convert them to CAISEvent objects  
3. Manage SessionTrace objects for rollouts
4. Save traces to disk for analysis
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

from opentelemetry import trace
from langfuse import Langfuse
from langfuse.openai import openai, AzureOpenAI
from synth_ai.tracing_v2.core import TraceHook, TraceStateHook
from synth_ai.tracing_v2.config import detect_provider

# Note: Anthropic support would require langfuse.anthropic when available
# For now, we can detect Anthropic models but the integration is through standard OTEL


@dataclass
class TimeRecord:
    """Time record for events and messages."""
    event_time: str  # Wall-clock timestamp
    message_time: int  # Turn/step number
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionEventMessage:
    """Message entering or leaving a Markov blanket."""
    content: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_type: str = "unknown"
    time_record: Optional[TimeRecord] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_record:
            result['time_record'] = self.time_record.to_dict()
        return result


# Base SessionEvent class
@dataclass
class SessionEvent:
    """Base class for session events (system state changes)."""
    time_record: Optional[TimeRecord] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_record:
            result['time_record'] = self.time_record.to_dict()
        return result


@dataclass  
class CAISEvent(SessionEvent):
    """CAIS system event capturing LLM interactions."""
    system_instance_id: str = "llm_agent"
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None
    
    # LLM call records - for storing complete LLM interaction details
    llm_call_records: List[Any] = field(default_factory=list)
    
    # OTEL/Langfuse specific fields
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_record:
            result['time_record'] = self.time_record.to_dict()
        return result


@dataclass
class RuntimeEvent(SessionEvent):
    """Runtime system event (e.g., action execution, message routing)."""
    system_instance_id: str = "runtime"
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None
    actions: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_record:
            result['time_record'] = self.time_record.to_dict()
        return result


@dataclass
class EnvironmentEvent(SessionEvent):
    """Environment system event (e.g., state transition, reward signal)."""
    system_instance_id: str = "environment"
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    observation_diff: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_record:
            result['time_record'] = self.time_record.to_dict()
        return result


@dataclass
class SessionTimeStep:
    """A single timestep in a session containing events and messages."""
    step_id: str
    events: List[SessionEvent] = field(default_factory=list)
    step_messages: List[SessionEventMessage] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    step_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: SessionEvent):
        """Add an event to this timestep."""
        self.events.append(event)
    
    def add_message(self, message: SessionEventMessage):
        """Add a message to this timestep."""
        self.step_messages.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['events'] = [event.to_dict() for event in self.events]
        result['step_messages'] = [msg.to_dict() for msg in self.step_messages]
        return result


@dataclass
class SessionMetadum:
    """Base class for session metadata."""
    metadata_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionTrace:
    """Complete trace of a session with timesteps and metadata."""
    session_id: str
    session_time_steps: List[SessionTimeStep] = field(default_factory=list)
    session_metadata: List[SessionMetadum] = field(default_factory=list)
    message_history: List[SessionEventMessage] = field(default_factory=list)
    event_history: List[SessionEvent] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_timestep(self, timestep: SessionTimeStep):
        """Add a timestep to the session."""
        self.session_time_steps.append(timestep)
    
    def add_metadata(self, metadata: SessionMetadum):
        """Add metadata to the session."""
        self.session_metadata.append(metadata)
    
    def add_message(self, message: SessionEventMessage):
        """Add a message to the global message history."""
        self.message_history.append(message)
    
    def add_event(self, event: SessionEvent):
        """Add an event to the global event history."""
        self.event_history.append(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['session_time_steps'] = [step.to_dict() for step in self.session_time_steps]
        result['session_metadata'] = [meta.to_dict() for meta in self.session_metadata]
        result['message_history'] = [msg.to_dict() for msg in self.message_history]
        result['event_history'] = [event.to_dict() for event in self.event_history]
        return result


def extract_model_info_from_attrs(attrs: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """Generic helper to extract model/token info for various providers."""
    return {
        "model_name": attrs.get("llm.model_name") or attrs.get("gen_ai.request.model"),
        "prompt_tokens": attrs.get("llm.usage.prompt_tokens") or attrs.get("gen_ai.usage.prompt_tokens"),
        "completion_tokens": attrs.get("llm.usage.completion_tokens") or attrs.get("gen_ai.usage.completion_tokens"),
        "total_tokens": attrs.get("llm.usage.total_tokens") or attrs.get("gen_ai.usage.total_tokens"),
        "latency_ms": attrs.get("llm.latency") or attrs.get("gen_ai.latency"),
        "cost": attrs.get("llm.cost") or attrs.get("gen_ai.cost"),
    }


def guess_provider_from_attrs(attrs: Dict[str, Any]) -> Optional[str]:
    """Guess the provider based on known attribute keys - supports OpenAI, Azure OpenAI, and Anthropic."""
    # First check if provider is explicitly set
    if "gen_ai.request.provider" in attrs:
        return attrs["gen_ai.request.provider"]
    
    # Extract model name and endpoint
    model_name = str(attrs.get("llm.model_name", "") or attrs.get("gen_ai.request.model", ""))
    endpoint = str(attrs.get("llm.endpoint", "") or attrs.get("gen_ai.endpoint", ""))
    
    # Use config to detect provider
    return detect_provider(model_name, endpoint)


class SessionTracer:
    """Main class for tracing LLM interactions in sessions."""
    
    def __init__(self, traces_dir: str = "traces", hooks: Optional[List[TraceHook]] = None, 
                 duckdb_path: Optional[str] = None, experiment_id: Optional[str] = None):
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(exist_ok=True)
        self.current_session: Optional[SessionTrace] = None
        self.current_timestep: Optional[SessionTimeStep] = None
        self.current_turn: int = 0
        self.hooks: List[TraceHook] = hooks or []
        self.experiment_id = experiment_id
        
        # Storage integration
        from .config import LOCAL_SYNTH, DUCKDB_CONFIG, SYNTH_CLOUD_CONFIG
        
        self.duckdb_path = duckdb_path
        self.db_manager = None
        
        # Check storage mode
        if not LOCAL_SYNTH and SYNTH_CLOUD_CONFIG["enabled"]:
            print("Warning: Synth Cloud storage not yet implemented. Using local storage.")
            # For now, fall back to local storage
            LOCAL_SYNTH = True
        
        # Initialize DuckDB if in local mode
        if LOCAL_SYNTH and (duckdb_path or DUCKDB_CONFIG["enabled"]):
            from .duckdb.manager import DuckDBTraceManager
            db_path = duckdb_path or DUCKDB_CONFIG["db_path"]
            self.db_manager = DuckDBTraceManager(db_path)
        
    def start_session(self, session_id: Optional[str] = None) -> SessionTrace:
        """Start a new session trace."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.current_session = SessionTrace(session_id=session_id)
        self.current_turn = 0
        return self.current_session
    
    def start_timestep(self, step_id: Optional[str] = None, **metadata) -> SessionTimeStep:
        """Start a new timestep in the current session."""
        if self.current_session is None:
            raise ValueError("Must start a session before creating timesteps")
        
        if step_id is None:
            step_id = f"step_{len(self.current_session.session_time_steps)}"
        
        self.current_timestep = SessionTimeStep(step_id=step_id, step_metadata=metadata)
        self.current_session.add_timestep(self.current_timestep)
        return self.current_timestep
    
    def record_message(self, message: SessionEventMessage) -> SessionEventMessage:
        """Record a message in both the current timestep and global history."""
        if self.current_session is None:
            raise ValueError("Must start a session before recording messages")
        if self.current_timestep is None:
            raise ValueError("Must start a timestep before recording messages")
        
        # Set time record if not already set
        if message.time_record is None:
            message.time_record = TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=self.current_turn
            )
        
        # Add to both local and global histories
        self.current_timestep.add_message(message)
        self.current_session.add_message(message)
        return message
    
    def _run_hooks_on_event(self, event: SessionEvent):
        """Run all registered hooks on an event and collect metadata."""
        if not self.hooks:
            return
        
        for hook in self.hooks:
            try:
                # Check if this hook applies to this event
                hook_result = hook.check(event)
                if hook_result and hasattr(event, 'event_metadata'):
                    # Convert HookResult to dict for storage
                    event.event_metadata.append(asdict(hook_result))
            except Exception as e:
                # Silently ignore hook errors in production
                pass
    
    def record_event(self, event: SessionEvent) -> SessionEvent:
        """Record an event in both the current timestep and global history."""
        if self.current_session is None:
            raise ValueError("Must start a session before recording events")
        if self.current_timestep is None:
            raise ValueError("Must start a timestep before recording events")
        
        # Set time record if not already set
        if event.time_record is None:
            event.time_record = TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=self.current_turn
            )
        
        # Run hooks to generate event metadata
        self._run_hooks_on_event(event)
        
        # Add to both local and global histories
        self.current_timestep.add_event(event)
        self.current_session.add_event(event)
        return event
    
    def capture_llm_call_from_generation(self,
                                       generation,
                                       system_id: str = "llm_agent",
                                       system_state_before: Optional[Dict] = None,
                                       system_state_after: Optional[Dict] = None,
                                       messages: Optional[List] = None,
                                       response = None) -> Optional[CAISEvent]:
        """
        Capture LLM call from langfuse generation object.
        Records prompt message, then completion message, then CAISEvent in temporal order.
        """
        if self.current_timestep is None:
            raise ValueError("Must start a timestep before capturing LLM calls")
        
        try:
            # Record inbound message (prompt) first
            if messages:
                inbound_message = SessionEventMessage(
                    content=messages,
                    message_type="llm_prompt",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=self.current_turn
                    )
                )
                self.record_message(inbound_message)
            
            # Create CAISEvent for the LLM call
            usage = getattr(response, 'usage', None) if response else None
            model_name = getattr(response, 'model', None) if response else None
            
            # Determine provider
            provider = None
            if hasattr(generation, 'provider'):
                provider = generation.provider
            elif model_name:
                # Try to get endpoint from generation metadata
                endpoint = ""
                if hasattr(generation, 'metadata') and isinstance(generation.metadata, dict):
                    endpoint = generation.metadata.get('endpoint', '')
                provider = detect_provider(model_name, endpoint)
            
            event = CAISEvent(
                system_instance_id=system_id,
                system_state_before=system_state_before,
                system_state_after=system_state_after,
                span_id=str(generation.id) if hasattr(generation, 'id') else None,
                trace_id=str(generation.trace_id) if hasattr(generation, 'trace_id') else None,
                model_name=model_name,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=self.current_turn
                ),
                metadata={
                    "provider": provider,
                    "generation_id": str(generation.id) if hasattr(generation, 'id') else None,
                    "trace_url": generation.get_trace_url() if hasattr(generation, 'get_trace_url') else None,
                    "langfuse_generation_methods": [attr for attr in dir(generation) if not attr.startswith('_')],
                }
            )
            
            # Record the event
            self.record_event(event)
            
            # Record outbound message (completion) last
            if response:
                outbound_content = None
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message'):
                        outbound_content = {
                            "content": choice.message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    },
                                    "type": tc.type
                                } for tc in choice.message.tool_calls
                            ] if choice.message.tool_calls else []
                        }
                
                if outbound_content:
                    outbound_message = SessionEventMessage(
                        content=outbound_content,
                        message_type="llm_completion",
                        time_record=TimeRecord(
                            event_time=datetime.now().isoformat(),
                            message_time=self.current_turn
                        )
                    )
                    self.record_message(outbound_message)
            
            return event
            
        except Exception as e:
            print(f"Warning: Failed to capture LLM call: {e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_llm_call(self, 
                        system_id: str = "llm_agent",
                        system_state_before: Optional[Dict] = None,
                        system_state_after: Optional[Dict] = None) -> Optional[CAISEvent]:
        """
        Capture the current langfuse OTEL span and convert to CAISEvent.
        Records prompt message, then CAISEvent, then completion message in temporal order.
        """
        if self.current_timestep is None:
            raise ValueError("Must start a timestep before capturing LLM calls")
        
        try:
            # Get current OTEL span from langfuse
            current_span = trace.get_current_span()
            
            if not current_span or not current_span.is_recording():
                return None
            
            # Extract span attributes
            attrs = dict(current_span.attributes) if current_span.attributes else {}
            
            # Record inbound message (prompt) first
            inbound_content = self._extract_prompt_from_attrs(attrs)
            if inbound_content:
                inbound_message = SessionEventMessage(
                    content=inbound_content,
                    message_type="llm_prompt",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=self.current_turn
                    )
                )
                self.record_message(inbound_message)
            
            # Extract model info using helper
            model_info = extract_model_info_from_attrs(attrs)
            
            # Guess provider
            provider = guess_provider_from_attrs(attrs)
            
            # Create and record CAIS event
            event = CAISEvent(
                system_instance_id=system_id,
                system_state_before=system_state_before,
                system_state_after=system_state_after,
                span_id=str(current_span.context.span_id) if current_span.context else None,
                trace_id=str(current_span.context.trace_id) if current_span.context else None,
                model_name=model_info["model_name"],
                prompt_tokens=model_info["prompt_tokens"],
                completion_tokens=model_info["completion_tokens"],
                total_tokens=model_info["total_tokens"],
                cost=model_info["cost"],
                latency_ms=model_info["latency_ms"],
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=self.current_turn
                ),
                metadata={
                    **attrs,
                    "provider": provider
                }
            )
            
            self.record_event(event)
            
            # Record outbound message (completion) last
            outbound_content = self._extract_completion_from_attrs(attrs)
            if outbound_content:
                outbound_message = SessionEventMessage(
                    content=outbound_content,
                    message_type="llm_completion",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=self.current_turn
                    )
                )
                self.record_message(outbound_message)
            
            return event
            
        except Exception as e:
            print(f"Warning: Failed to capture LLM call: {e}")
            return None
    
    def _extract_prompt_from_attrs(self, attrs: Dict) -> Optional[Any]:
        """Extract prompt content from OTEL attributes."""
        # Try different possible attribute names
        for key in ["llm.prompts", "gen_ai.prompt", "llm.input_messages", "messages"]:
            if key in attrs:
                return attrs[key]
        return None
    
    def _extract_completion_from_attrs(self, attrs: Dict) -> Optional[Any]:
        """Extract completion content from OTEL attributes."""
        # Try different possible attribute names  
        for key in ["llm.completions", "gen_ai.completion", "llm.output_messages", "completion"]:
            if key in attrs:
                return attrs[key]
        return None
    
    def add_session_metadata(self, metadata_type: str, data: Dict[str, Any]):
        """Add metadata to the current session."""
        if self.current_session is None:
            raise ValueError("Must start a session before adding metadata")
        
        metadata = SessionMetadum(metadata_type=metadata_type, data=data)
        self.current_session.add_metadata(metadata)
    
    def advance_turn(self):
        """Advance to the next turn."""
        self.current_turn += 1
    
    def save_session(self, filename: Optional[str] = None) -> Path:
        """Save the current session trace to disk."""
        if self.current_session is None:
            raise ValueError("No active session to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{self.current_session.session_id}_{timestamp}.json"
        
        filepath = self.traces_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2)
        
        # Don't print the filepath as it messes with progress bar display
        # print(f"📁 Saved session trace: {filepath}")
        return filepath
    
    def end_session(self, save: bool = True, upload_to_db: bool = True) -> Optional[Path]:
        """End the current session and optionally save it."""
        if self.current_session is None:
            return None
        
        # Upload to DuckDB if enabled
        if self.db_manager and upload_to_db:
            try:
                self.db_manager.insert_session_trace(self.current_session)
                
                # Link to experiment if specified
                if self.experiment_id:
                    self.db_manager.link_session_to_experiment(
                        self.current_session.session_id, 
                        self.experiment_id
                    )
            except Exception as e:
                print(f"Failed to upload trace to DuckDB: {e}")
                import traceback
                traceback.print_exc()
        
        filepath = None
        if save:
            filepath = self.save_session()
        
        self.current_session = None
        self.current_timestep = None
        self.current_turn = 0
        return filepath
    
    def close(self):
        """Close resources including DuckDB connection."""
        if self.db_manager:
            self.db_manager.close()
            self.db_manager = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for simple usage
def create_session_tracer(traces_dir: str = "traces", hooks: Optional[List[TraceHook]] = None) -> SessionTracer:
    """Create a new session tracer."""
    return SessionTracer(traces_dir, hooks=hooks) 