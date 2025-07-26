"""
Tests for v2 tracing serialization functionality.

This module tests:
- Serialization of all v2 tracing dataclasses
- to_dict() methods
- JSON compatibility
- UUID and datetime handling
"""

import pytest
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

from synth_ai.tracing_v2.abstractions import (
    SessionEvent, SessionMessage, SessionTimeStep, SessionTrace,
    SystemEvent, CAISEvent, EnvironmentEvent, RuntimeEvent,
    SessionEventMessage, TimeRecord
)
from synth_ai.tracing_v2.utils import make_serializable


class TestMakeSerializable:
    """Test the make_serializable utility function."""
    
    def test_primitive_types(self):
        """Test that primitive types pass through unchanged."""
        assert make_serializable(42) == 42
        assert make_serializable(3.14) == 3.14
        assert make_serializable("hello") == "hello"
        assert make_serializable(True) is True
        assert make_serializable(None) is None
    
    def test_uuid_serialization(self):
        """Test UUID serialization."""
        test_uuid = uuid.uuid4()
        result = make_serializable(test_uuid)
        assert isinstance(result, str)
        assert result == str(test_uuid)
    
    def test_datetime_serialization(self):
        """Test datetime serialization."""
        test_datetime = datetime.now()
        result = make_serializable(test_datetime)
        assert isinstance(result, str)
        assert result == test_datetime.isoformat()
    
    def test_list_serialization(self):
        """Test list serialization with mixed types."""
        test_list = [
            "string",
            42,
            uuid.uuid4(),
            datetime.now(),
            {"nested": "dict"}
        ]
        result = make_serializable(test_list)
        
        assert isinstance(result, list)
        assert len(result) == len(test_list)
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)
        assert isinstance(result[2], str)  # UUID -> str
        assert isinstance(result[3], str)  # datetime -> str
        assert isinstance(result[4], dict)
    
    def test_dict_serialization(self):
        """Test dictionary serialization with nested structures."""
        test_dict = {
            "id": uuid.uuid4(),
            "timestamp": datetime.now(),
            "data": {
                "values": [1, 2, 3],
                "metadata": {
                    "created": datetime.now(),
                    "session_id": uuid.uuid4()
                }
            },
            "simple": "value"
        }
        
        result = make_serializable(test_dict)
        
        assert isinstance(result, dict)
        assert isinstance(result["id"], str)
        assert isinstance(result["timestamp"], str)
        assert isinstance(result["data"]["metadata"]["created"], str)
        assert isinstance(result["data"]["metadata"]["session_id"], str)
        assert result["simple"] == "value"
    
    def test_custom_object_with_to_dict(self):
        """Test objects with to_dict method."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
                self.id = uuid.uuid4()
            
            def to_dict(self):
                return {
                    "value": self.value,
                    "id": str(self.id)
                }
        
        obj = CustomObject("test")
        result = make_serializable(obj)
        
        assert isinstance(result, dict)
        assert result["value"] == "test"
        assert isinstance(result["id"], str)
    
    def test_unsupported_type(self):
        """Test handling of unsupported types."""
        class UnsupportedType:
            pass
        
        obj = UnsupportedType()
        result = make_serializable(obj)
        
        # Should return string representation
        assert isinstance(result, str)
        assert "UnsupportedType" in result


class TestTimeRecordSerialization:
    """Test TimeRecord serialization."""
    
    def test_time_record_creation_and_serialization(self):
        """Test creating and serializing TimeRecord."""
        event_time = datetime.now().isoformat()
        message_time = 42
        
        record = TimeRecord(event_time=event_time, message_time=message_time)
        
        # Test to_dict
        result = record.to_dict()
        assert isinstance(result, dict)
        assert result["event_time"] == event_time
        assert result["message_time"] == message_time
        
        # Test JSON serialization
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded["event_time"] == event_time
        assert loaded["message_time"] == message_time


class TestSessionEventMessageSerialization:
    """Test SessionEventMessage serialization."""
    
    def test_basic_message_serialization(self):
        """Test basic message serialization."""
        message = SessionEventMessage(
            content={"action": "move", "direction": "north"},
            message_type="agent_action"
        )
        
        result = message.to_dict()
        
        assert isinstance(result, dict)
        assert result["content"] == {"action": "move", "direction": "north"}
        assert result["message_type"] == "agent_action"
        assert "timestamp" in result
        assert result["time_record"] is None
    
    def test_message_with_time_record(self):
        """Test message with TimeRecord serialization."""
        time_record = TimeRecord(
            event_time=datetime.now().isoformat(),
            message_time=5
        )
        
        message = SessionEventMessage(
            content={"observation": "You see a door"},
            message_type="environment",
            time_record=time_record
        )
        
        result = message.to_dict()
        
        assert isinstance(result["time_record"], dict)
        assert result["time_record"]["message_time"] == 5
        
        # Test JSON compatibility
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded["time_record"]["message_time"] == 5
    
    def test_message_with_complex_content(self):
        """Test message with complex nested content."""
        content = {
            "origin_system_id": str(uuid.uuid4()),
            "payload": {
                "observation": {
                    "position": [10, 20],
                    "inventory": {"wood": 5, "stone": 3},
                    "timestamp": datetime.now()
                },
                "metadata": {
                    "session_id": uuid.uuid4(),
                    "turn": 10
                }
            }
        }
        
        message = SessionEventMessage(
            content=content,
            message_type="complex_observation"
        )
        
        result = message.to_dict()
        
        # Verify structure is preserved
        assert isinstance(result["content"]["origin_system_id"], str)
        assert result["content"]["payload"]["observation"]["position"] == [10, 20]
        assert isinstance(result["content"]["payload"]["observation"]["timestamp"], str)
        assert isinstance(result["content"]["payload"]["metadata"]["session_id"], str)
        
        # Verify JSON serializable
        json.dumps(result)  # Should not raise


class TestCAISEventSerialization:
    """Test CAISEvent serialization."""
    
    def test_minimal_cais_event(self):
        """Test minimal CAISEvent serialization."""
        event = CAISEvent(
            system_instance_id="test_agent"
        )
        
        result = event.to_dict()
        
        assert result["system_instance_id"] == "test_agent"
        assert result["span_id"] is None
        assert result["trace_id"] is None
        assert result["model_name"] is None
        
        # Should be JSON serializable
        json.dumps(result)
    
    def test_complete_cais_event(self):
        """Test complete CAISEvent with all fields."""
        event = CAISEvent(
            system_instance_id="llm_agent",
            span_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cost=0.015,
            latency_ms=1234.56,
            system_state_before={
                "temperature": 0.7,
                "max_tokens": 1000,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
            },
            system_state_after={
                "response": "Hi there!",
                "finish_reason": "stop"
            },
            llm_call_records=[{
                "timestamp": datetime.now(),
                "duration_ms": 1234.56,
                "model": "gpt-4"
            }],
            metadata={
                "session_id": uuid.uuid4(),
                "experiment": "test_exp"
            },
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=10
            )
        )
        
        result = event.to_dict()
        
        # Check all fields are present and properly serialized
        assert result["system_instance_id"] == "llm_agent"
        assert isinstance(result["span_id"], str)
        assert isinstance(result["trace_id"], str)
        assert result["model_name"] == "gpt-4"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 200
        assert result["total_tokens"] == 300
        assert result["cost"] == 0.015
        assert result["latency_ms"] == 1234.56
        
        # Check complex fields
        assert result["system_state_before"]["temperature"] == 0.7
        assert len(result["system_state_before"]["messages"]) == 2
        assert result["system_state_after"]["response"] == "Hi there!"
        
        # Check serialization of nested objects
        assert isinstance(result["llm_call_records"][0]["timestamp"], str)
        assert isinstance(result["metadata"]["session_id"], str)
        assert isinstance(result["time_record"], dict)
        
        # Verify JSON serializable
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded["model_name"] == "gpt-4"


class TestEnvironmentEventSerialization:
    """Test EnvironmentEvent serialization."""
    
    def test_environment_event_serialization(self):
        """Test EnvironmentEvent serialization."""
        event = EnvironmentEvent(
            environment_name="crafter",
            state_before={
                "position": [0, 0],
                "health": 100,
                "timestamp": datetime.now()
            },
            state_after={
                "position": [1, 0],
                "health": 90,
                "timestamp": datetime.now()
            },
            action_taken={"type": "move", "direction": "east"},
            reward=-10.0,
            metadata={
                "step": 1,
                "episode_id": uuid.uuid4()
            },
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            )
        )
        
        result = event.to_dict()
        
        assert result["environment_name"] == "crafter"
        assert result["state_before"]["position"] == [0, 0]
        assert result["state_after"]["health"] == 90
        assert result["action_taken"]["type"] == "move"
        assert result["reward"] == -10.0
        
        # Check datetime and UUID serialization
        assert isinstance(result["state_before"]["timestamp"], str)
        assert isinstance(result["state_after"]["timestamp"], str)
        assert isinstance(result["metadata"]["episode_id"], str)
        
        # Verify JSON serializable
        json.dumps(result)


class TestRuntimeEventSerialization:
    """Test RuntimeEvent serialization."""
    
    def test_runtime_event_serialization(self):
        """Test RuntimeEvent serialization."""
        event = RuntimeEvent(
            event_type="function_call",
            function_name="process_data",
            duration_ms=123.45,
            success=True,
            error_message=None,
            metadata={
                "input_size": 1024,
                "output_size": 2048,
                "timestamp": datetime.now(),
                "trace_id": uuid.uuid4()
            },
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=5
            )
        )
        
        result = event.to_dict()
        
        assert result["event_type"] == "function_call"
        assert result["function_name"] == "process_data"
        assert result["duration_ms"] == 123.45
        assert result["success"] is True
        assert result["error_message"] is None
        
        # Check metadata serialization
        assert result["metadata"]["input_size"] == 1024
        assert isinstance(result["metadata"]["timestamp"], str)
        assert isinstance(result["metadata"]["trace_id"], str)
        
        # Verify JSON serializable
        json.dumps(result)
    
    def test_runtime_event_with_error(self):
        """Test RuntimeEvent with error information."""
        event = RuntimeEvent(
            event_type="api_call",
            function_name="external_api",
            duration_ms=5000.0,
            success=False,
            error_message="Connection timeout",
            metadata={
                "url": "https://api.example.com",
                "retry_count": 3
            }
        )
        
        result = event.to_dict()
        
        assert result["success"] is False
        assert result["error_message"] == "Connection timeout"
        assert result["metadata"]["retry_count"] == 3


class TestSessionTimeStepSerialization:
    """Test SessionTimeStep serialization."""
    
    def test_timestep_serialization(self):
        """Test SessionTimeStep serialization."""
        messages = [
            SessionEventMessage(
                content={"action": "move"},
                message_type="agent",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=1
                )
            ),
            SessionEventMessage(
                content={"observation": "moved"},
                message_type="environment",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=1
                )
            )
        ]
        
        events = [
            CAISEvent(
                system_instance_id="agent",
                model_name="gpt-4",
                prompt_tokens=10,
                completion_tokens=20
            )
        ]
        
        timestep = SessionTimeStep(
            turn_number=1,
            messages=messages,
            events=events
        )
        
        result = timestep.to_dict()
        
        assert result["turn_number"] == 1
        assert len(result["messages"]) == 2
        assert len(result["events"]) == 1
        
        # Check message serialization
        assert result["messages"][0]["message_type"] == "agent"
        assert result["messages"][1]["message_type"] == "environment"
        
        # Check event serialization
        assert result["events"][0]["model_name"] == "gpt-4"
        
        # Verify JSON serializable
        json.dumps(result)


class TestSessionTraceSerialization:
    """Test complete SessionTrace serialization."""
    
    def test_complete_session_trace(self):
        """Test serialization of a complete session trace."""
        # Create timesteps
        timesteps = []
        for i in range(3):
            messages = [
                SessionEventMessage(
                    content={
                        "origin_system_id": f"agent_{i}",
                        "payload": {"action": f"action_{i}"}
                    },
                    message_type="action",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=i
                    )
                )
            ]
            
            events = [
                CAISEvent(
                    system_instance_id=f"agent_{i}",
                    model_name="gpt-4",
                    prompt_tokens=10 * (i + 1),
                    completion_tokens=20 * (i + 1),
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=i
                    )
                )
            ]
            
            timestep = SessionTimeStep(
                turn_number=i,
                messages=messages,
                events=events
            )
            timesteps.append(timestep)
        
        # Create event history
        event_history = [
            CAISEvent(
                system_instance_id="global_agent",
                model_name="gpt-4",
                total_tokens=150
            ),
            EnvironmentEvent(
                environment_name="test_env",
                state_before={"health": 100},
                state_after={"health": 90},
                action_taken={"type": "damage"},
                reward=-10.0
            )
        ]
        
        # Create session trace
        session_id = str(uuid.uuid4())
        trace = SessionTrace(
            session_id=session_id,
            metadata={
                "experiment": "test",
                "version": "1.0",
                "start_time": datetime.now(),
                "config": {
                    "temperature": 0.7,
                    "max_turns": 10
                }
            },
            timesteps=timesteps,
            message_history=[],  # Could be populated
            event_history=event_history
        )
        
        # Serialize
        result = trace.to_dict()
        
        # Verify structure
        assert result["session_id"] == session_id
        assert result["metadata"]["experiment"] == "test"
        assert isinstance(result["metadata"]["start_time"], str)
        
        # Verify timesteps
        assert len(result["timesteps"]) == 3
        for i, ts in enumerate(result["timesteps"]):
            assert ts["turn_number"] == i
            assert len(ts["messages"]) == 1
            assert len(ts["events"]) == 1
        
        # Verify event history
        assert len(result["event_history"]) == 2
        assert result["event_history"][0]["system_instance_id"] == "global_agent"
        assert result["event_history"][1]["environment_name"] == "test_env"
        
        # Verify JSON serializable
        json_str = json.dumps(result, indent=2)
        
        # Verify round-trip
        loaded = json.loads(json_str)
        assert loaded["session_id"] == session_id
        assert len(loaded["timesteps"]) == 3
        assert len(loaded["event_history"]) == 2
    
    def test_empty_session_trace(self):
        """Test serialization of empty session trace."""
        trace = SessionTrace(
            session_id="empty_session",
            metadata={},
            timesteps=[],
            message_history=[],
            event_history=[]
        )
        
        result = trace.to_dict()
        
        assert result["session_id"] == "empty_session"
        assert result["metadata"] == {}
        assert result["timesteps"] == []
        assert result["message_history"] == []
        assert result["event_history"] == []
        
        # Should still be JSON serializable
        json.dumps(result)