"""
Tests for basic v2 tracing functionality.

This module tests the core components of the v2 tracing system:
- SessionTracer functionality
- Event creation and management
- Message handling
- Time records
"""

import pytest
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord, CAISEvent, 
    RuntimeEvent, EnvironmentEvent, SessionTrace
)
from synth_ai.tracing_v2.abstractions import (
    SessionEvent, SessionMessage
)
from synth_ai.tracing_v2.hooks import TraceHook, TraceStateHook


class TestTimeRecord:
    """Test the TimeRecord class."""
    
    def test_time_record_creation(self):
        """Test creating a TimeRecord instance."""
        event_time = datetime.now().isoformat()
        message_time = 5
        
        record = TimeRecord(event_time=event_time, message_time=message_time)
        
        assert record.event_time == event_time
        assert record.message_time == message_time
    
    def test_time_record_to_dict(self):
        """Test converting TimeRecord to dictionary."""
        event_time = datetime.now().isoformat()
        message_time = 10
        
        record = TimeRecord(event_time=event_time, message_time=message_time)
        result = record.to_dict()
        
        assert result["event_time"] == event_time
        assert result["message_time"] == message_time
        assert len(result) == 2


class TestSessionEventMessage:
    """Test the SessionEventMessage class."""
    
    def test_message_creation(self):
        """Test creating a SessionEventMessage instance."""
        content = {"action": "move", "direction": "north"}
        message_type = "agent_action"
        
        message = SessionEventMessage(
            content=content,
            message_type=message_type
        )
        
        assert message.content == content
        assert message.message_type == message_type
        assert message.timestamp is not None
        assert message.time_record is None
    
    def test_message_with_time_record(self):
        """Test creating a message with TimeRecord."""
        content = {"observation": "You see a tree"}
        message_type = "environment_observation"
        time_record = TimeRecord(
            event_time=datetime.now().isoformat(),
            message_time=3
        )
        
        message = SessionEventMessage(
            content=content,
            message_type=message_type,
            time_record=time_record
        )
        
        assert message.time_record == time_record
        assert message.time_record.message_time == 3
    
    def test_message_to_dict(self):
        """Test converting SessionEventMessage to dictionary."""
        content = {"test": "data"}
        message = SessionEventMessage(
            content=content,
            message_type="test_type",
            time_record=TimeRecord(
                event_time="2024-01-01T00:00:00",
                message_time=1
            )
        )
        
        result = message.to_dict()
        
        assert result["content"] == content
        assert result["message_type"] == "test_type"
        assert "timestamp" in result
        assert "time_record" in result
        assert result["time_record"]["message_time"] == 1


class TestCAISEvent:
    """Test the CAISEvent class."""
    
    def test_cais_event_creation(self):
        """Test creating a CAISEvent instance."""
        event = CAISEvent(
            system_instance_id="test_agent",
            span_id="span123",
            trace_id="trace456",
            model_name="gpt-4",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            cost=0.005,
            latency_ms=1200.5
        )
        
        assert event.system_instance_id == "test_agent"
        assert event.span_id == "span123"
        assert event.trace_id == "trace456"
        assert event.model_name == "gpt-4"
        assert event.prompt_tokens == 50
        assert event.completion_tokens == 100
        assert event.total_tokens == 150
        assert event.cost == 0.005
        assert event.latency_ms == 1200.5
    
    def test_cais_event_with_llm_records(self):
        """Test CAISEvent with LLM call records."""
        llm_record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ],
            "response": {"role": "assistant", "content": "Hi there!"},
            "timestamp": datetime.now().isoformat()
        }
        
        event = CAISEvent(
            system_instance_id="llm_agent",
            llm_call_records=[llm_record],
            metadata={"session_id": "test123"}
        )
        
        assert len(event.llm_call_records) == 1
        assert event.llm_call_records[0] == llm_record
        assert event.metadata["session_id"] == "test123"
    
    def test_cais_event_to_dict(self):
        """Test converting CAISEvent to dictionary."""
        event = CAISEvent(
            system_instance_id="test",
            model_name="claude-3",
            time_record=TimeRecord(
                event_time="2024-01-01T00:00:00",
                message_time=5
            )
        )
        
        result = event.to_dict()
        
        assert result["system_instance_id"] == "test"
        assert result["model_name"] == "claude-3"
        assert "time_record" in result
        assert result["time_record"]["message_time"] == 5


class TestSessionTracer:
    """Test the SessionTracer class."""
    
    @pytest.fixture
    def tracer(self):
        """Create a SessionTracer instance for testing."""
        return SessionTracer()
    
    @pytest.mark.asyncio
    async def test_start_session(self, tracer):
        """Test starting a new session."""
        session_id = str(uuid.uuid4())
        metadata = {"experiment": "test", "version": "1.0"}
        
        async with tracer.start_session(session_id, metadata) as session:
            assert tracer.current_session is not None
            assert tracer.current_session.session_id == session_id
            assert tracer.current_session.metadata == metadata
            assert len(tracer._sessions) == 1
            assert session_id in tracer._sessions
        
        # After context exit, current_session should be None
        assert tracer.current_session is None
    
    @pytest.mark.asyncio
    async def test_timestep_management(self, tracer):
        """Test timestep creation and management."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            # Test timestep 1
            async with tracer.timestep(1) as ts1:
                assert tracer.current_timestep is not None
                assert tracer.current_timestep.turn_number == 1
                assert tracer.in_timestep is True
            
            # After timestep exit
            assert tracer.current_timestep is None
            assert tracer.in_timestep is False
            
            # Test timestep 2
            async with tracer.timestep(2) as ts2:
                assert tracer.current_timestep.turn_number == 2
    
    @pytest.mark.asyncio
    async def test_add_message(self, tracer):
        """Test adding messages to timesteps."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            async with tracer.timestep(1):
                message = SessionEventMessage(
                    content={"action": "move"},
                    message_type="agent_action",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=1
                    )
                )
                
                await tracer.add_message(message)
                
                # Verify message was added
                assert len(tracer.current_timestep.messages) == 1
                assert tracer.current_timestep.messages[0] == message
    
    @pytest.mark.asyncio
    async def test_add_event(self, tracer):
        """Test adding events to session."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            event = CAISEvent(
                system_instance_id="test_agent",
                model_name="gpt-4",
                prompt_tokens=10,
                completion_tokens=20
            )
            
            await tracer.add_event(event)
            
            # Verify event was added
            session = tracer._sessions[session_id]
            assert len(session.event_history) == 1
            assert session.event_history[0] == event
    
    @pytest.mark.asyncio
    async def test_multiple_messages_in_timestep(self, tracer):
        """Test adding multiple messages within a single timestep."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            async with tracer.timestep(1):
                # Add agent message
                agent_msg = SessionEventMessage(
                    content={"action": "look"},
                    message_type="agent_action"
                )
                await tracer.add_message(agent_msg)
                
                # Add environment message
                env_msg = SessionEventMessage(
                    content={"observation": "You see a door"},
                    message_type="environment_observation"
                )
                await tracer.add_message(env_msg)
                
                assert len(tracer.current_timestep.messages) == 2
                assert tracer.current_timestep.messages[0].message_type == "agent_action"
                assert tracer.current_timestep.messages[1].message_type == "environment_observation"
    
    @pytest.mark.asyncio
    async def test_session_without_timestep(self, tracer):
        """Test that messages require an active timestep."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            message = SessionEventMessage(
                content={"test": "data"},
                message_type="test"
            )
            
            # Should handle gracefully when no timestep is active
            await tracer.add_message(message)
            # No assertion needed - just ensure no exception
    
    @pytest.mark.asyncio
    async def test_get_session_trace(self, tracer):
        """Test retrieving a complete session trace."""
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id, {"test": "metadata"}):
            # Add some timesteps and messages
            async with tracer.timestep(1):
                await tracer.add_message(SessionEventMessage(
                    content={"action": "start"},
                    message_type="agent"
                ))
            
            async with tracer.timestep(2):
                await tracer.add_message(SessionEventMessage(
                    content={"action": "continue"},
                    message_type="agent"
                ))
            
            # Add an event
            await tracer.add_event(CAISEvent(
                system_instance_id="test",
                model_name="gpt-4"
            ))
        
        # Get the trace
        trace = tracer.get_session_trace(session_id)
        
        assert trace is not None
        assert trace.session_id == session_id
        assert trace.metadata["test"] == "metadata"
        assert len(trace.timesteps) == 2
        assert trace.timesteps[0].turn_number == 1
        assert trace.timesteps[1].turn_number == 2
        assert len(trace.event_history) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, tracer):
        """Test that tracer can handle multiple sessions (though not concurrently active)."""
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        
        # First session
        async with tracer.start_session(session1_id):
            async with tracer.timestep(1):
                await tracer.add_message(SessionEventMessage(
                    content={"session": 1},
                    message_type="test"
                ))
        
        # Second session
        async with tracer.start_session(session2_id):
            async with tracer.timestep(1):
                await tracer.add_message(SessionEventMessage(
                    content={"session": 2},
                    message_type="test"
                ))
        
        # Both sessions should be stored
        assert len(tracer._sessions) == 2
        assert session1_id in tracer._sessions
        assert session2_id in tracer._sessions
        
        # Verify content
        trace1 = tracer.get_session_trace(session1_id)
        trace2 = tracer.get_session_trace(session2_id)
        
        assert trace1.timesteps[0].messages[0].content["session"] == 1
        assert trace2.timesteps[0].messages[0].content["session"] == 2


class TestEnvironmentEvent:
    """Test the EnvironmentEvent class."""
    
    def test_environment_event_creation(self):
        """Test creating an EnvironmentEvent."""
        event = EnvironmentEvent(
            environment_name="crafter",
            state_before={"health": 100},
            state_after={"health": 90},
            action_taken={"type": "move", "direction": "north"},
            reward=-10,
            metadata={"step": 1}
        )
        
        assert event.environment_name == "crafter"
        assert event.state_before["health"] == 100
        assert event.state_after["health"] == 90
        assert event.action_taken["type"] == "move"
        assert event.reward == -10
        assert event.metadata["step"] == 1


class TestRuntimeEvent:
    """Test the RuntimeEvent class."""
    
    def test_runtime_event_creation(self):
        """Test creating a RuntimeEvent."""
        event = RuntimeEvent(
            event_type="function_call",
            function_name="process_action",
            duration_ms=45.2,
            success=True,
            error_message=None,
            metadata={"input_size": 1024}
        )
        
        assert event.event_type == "function_call"
        assert event.function_name == "process_action"
        assert event.duration_ms == 45.2
        assert event.success is True
        assert event.error_message is None
        assert event.metadata["input_size"] == 1024


class TestTraceHooks:
    """Test trace hook functionality."""
    
    @pytest.mark.asyncio
    async def test_trace_hook_integration(self):
        """Test that trace hooks are called during tracing."""
        hook_called = False
        hook_data = {}
        
        class TestHook(TraceHook):
            async def on_event(self, event: SessionEvent, session_id: str):
                nonlocal hook_called, hook_data
                hook_called = True
                hook_data["event"] = event
                hook_data["session_id"] = session_id
        
        tracer = SessionTracer()
        hook = TestHook()
        tracer.add_hook(hook)
        
        session_id = str(uuid.uuid4())
        async with tracer.start_session(session_id):
            event = CAISEvent(
                system_instance_id="test",
                model_name="gpt-4"
            )
            await tracer.add_event(event)
        
        # Give async operations time to complete
        await asyncio.sleep(0.1)
        
        assert hook_called
        assert hook_data["session_id"] == session_id
        assert isinstance(hook_data["event"], CAISEvent)
    
    @pytest.mark.asyncio
    async def test_state_hook_integration(self):
        """Test that state hooks can modify tracer state."""
        class TestStateHook(TraceStateHook):
            async def on_state_change(self, tracer: SessionTracer, state_type: str, **kwargs):
                if state_type == "session_start":
                    # Modify metadata
                    tracer.current_session.metadata["hook_added"] = True
        
        tracer = SessionTracer()
        hook = TestStateHook()
        tracer.add_hook(hook)
        
        session_id = str(uuid.uuid4())
        async with tracer.start_session(session_id, {"original": "data"}):
            session = tracer.current_session
            assert "hook_added" in session.metadata
            assert session.metadata["hook_added"] is True
            assert session.metadata["original"] == "data"


class TestSessionTraceSerialization:
    """Test serialization of session traces."""
    
    @pytest.mark.asyncio
    async def test_save_and_load_trace(self, tmp_path):
        """Test saving and loading a session trace."""
        tracer = SessionTracer()
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id, {"experiment": "test"}):
            async with tracer.timestep(1):
                await tracer.add_message(SessionEventMessage(
                    content={"action": "test"},
                    message_type="agent"
                ))
            
            await tracer.add_event(CAISEvent(
                system_instance_id="test",
                model_name="gpt-4",
                prompt_tokens=10
            ))
        
        # Save trace
        trace_file = tmp_path / f"{session_id}.json"
        tracer.save_trace(session_id, str(trace_file))
        
        assert trace_file.exists()
        
        # Load and verify
        import json
        with open(trace_file) as f:
            loaded_data = json.load(f)
        
        assert loaded_data["session_id"] == session_id
        assert loaded_data["metadata"]["experiment"] == "test"
        assert len(loaded_data["timesteps"]) == 1
        assert len(loaded_data["event_history"]) == 1