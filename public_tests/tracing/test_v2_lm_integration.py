"""
Tests for LM class integration with v2 tracing.

This module tests:
- LM class v2 tracing support
- Native tracing without modifying provider wrappers
- Trace capture and structure
- Context propagation through LM calls
"""

import pytest
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord, CAISEvent
)
from synth_ai.tracing_v2.decorators import (
    set_session_tracer, set_system_id, set_turn_number
)


class TestLMV2TracingIntegration:
    """Test LM class integration with v2 tracing."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm a helpful assistant."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello from Claude!"
            }],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25
            }
        }
    
    def test_lm_init_with_v2_tracing(self):
        """Test LM initialization with v2 tracing enabled."""
        tracer = SessionTracer()
        lm = LM(
            session_tracer=tracer,
            system_id="test_agent",
            enable_v2_tracing=True
        )
        
        assert lm._session_tracer == tracer
        assert lm._system_id == "test_agent"
        assert lm._enable_v2_tracing is True
    
    def test_lm_init_without_v2_tracing(self):
        """Test LM initialization with v2 tracing disabled."""
        lm = LM(enable_v2_tracing=False)
        
        assert lm._session_tracer is None
        assert lm._system_id is None
        assert lm._enable_v2_tracing is False
    
    @pytest.mark.asyncio
    async def test_lm_openai_call_with_tracing(self, mock_openai_response):
        """Test LM OpenAI call with v2 tracing enabled."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="test_openai_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("test_session"):
                # Set context for decorators
                set_session_tracer(tracer)
                set_system_id("test_openai_agent")
                set_turn_number(1)
                
                response = await lm.agenerate("Hello, world!")
        
        # Verify response
        assert response.text == "Hello! I'm a helpful assistant."
        
        # Verify event was captured
        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.system_instance_id == "test_openai_agent"
        assert event.model_name == "gpt-4"
        assert event.prompt_tokens == 10
        assert event.completion_tokens == 20
        assert event.total_tokens == 30
    
    @pytest.mark.asyncio
    async def test_lm_anthropic_call_with_tracing(self, mock_anthropic_response):
        """Test LM Anthropic call with v2 tracing enabled."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the Anthropic client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_anthropic_response)
        mock_client.messages.create = mock_create
        
        lm = LM(
            provider="anthropic",
            model="claude-3-opus-20240229",
            session_tracer=tracer,
            system_id="test_claude_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("test_session"):
                # Set context for decorators
                set_session_tracer(tracer)
                set_system_id("test_claude_agent")
                set_turn_number(1)
                
                response = await lm.agenerate("Hello from test!")
        
        # Verify response
        assert response.text == "Hello from Claude!"
        
        # Verify event was captured
        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.system_instance_id == "test_claude_agent"
        assert event.model_name == "claude-3-opus-20240229"
        assert event.prompt_tokens == 15
        assert event.completion_tokens == 25
        assert event.total_tokens == 40
    
    @pytest.mark.asyncio
    async def test_lm_tracing_disabled(self, mock_openai_response):
        """Test LM calls with v2 tracing disabled."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            enable_v2_tracing=False  # Tracing disabled
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            response = await lm.agenerate("Hello, world!")
        
        # Verify response
        assert response.text == "Hello! I'm a helpful assistant."
        
        # Verify no events were captured
        assert len(events_captured) == 0
    
    @pytest.mark.asyncio
    async def test_lm_with_system_state(self, mock_openai_response):
        """Test LM tracing captures system state correctly."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            session_tracer=tracer,
            system_id="test_state_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("test_session"):
                # Set context
                set_session_tracer(tracer)
                set_system_id("test_state_agent")
                set_turn_number(1)
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ]
                response = await lm.agenerate(messages)
        
        # Verify event system state
        assert len(events_captured) == 1
        event = events_captured[0]
        
        # Check system_state_before
        assert "gen_ai.request.model" in event.system_state_before
        assert event.system_state_before["gen_ai.request.model"] == "gpt-4"
        assert "gen_ai.request.temperature" in event.system_state_before
        assert event.system_state_before["gen_ai.request.temperature"] == 0.7
        assert "gen_ai.request.max_tokens" in event.system_state_before
        assert event.system_state_before["gen_ai.request.max_tokens"] == 100
        assert "gen_ai.request.messages" in event.system_state_before
        assert len(event.system_state_before["gen_ai.request.messages"]) == 2
        
        # Check system_state_after
        assert "gen_ai.response.content" in event.system_state_after
        assert event.system_state_after["gen_ai.response.content"] == "Hello! I'm a helpful assistant."
        assert "gen_ai.response.finish_reason" in event.system_state_after
        assert event.system_state_after["gen_ai.response.finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_lm_error_handling_with_tracing(self):
        """Test LM error handling with v2 tracing."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock client that raises an error
        mock_client = Mock()
        mock_create = AsyncMock(side_effect=Exception("API Error"))
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="test_error_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("test_session"):
                # Set context
                set_session_tracer(tracer)
                set_system_id("test_error_agent")
                set_turn_number(1)
                
                with pytest.raises(Exception, match="API Error"):
                    await lm.agenerate("Hello, world!")
        
        # Verify event was still captured with error info
        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.system_instance_id == "test_error_agent"
        # Error information should be in metadata
        assert "error" in event.metadata
        assert "API Error" in str(event.metadata["error"])
    
    @pytest.mark.asyncio
    async def test_lm_streaming_with_tracing(self, mock_openai_response):
        """Test LM streaming calls with v2 tracing."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock streaming response
        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {"content": "!"}}]},
                {"choices": [{"finish_reason": "stop"}], "usage": mock_openai_response["usage"]}
            ]
            for chunk in chunks:
                yield chunk
        
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_stream())
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="test_stream_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("test_session"):
                # Set context
                set_session_tracer(tracer)
                set_system_id("test_stream_agent")
                set_turn_number(1)
                
                # Collect streamed content
                content = ""
                async for chunk in lm.astream("Hello!"):
                    content += chunk.text
        
        assert content == "Hello world!"
        
        # Verify event was captured for streaming
        assert len(events_captured) >= 1
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.system_instance_id == "test_stream_agent"
        assert event.model_name == "gpt-4"
    
    def test_lm_sync_call_with_tracing(self, mock_openai_response):
        """Test synchronous LM calls with v2 tracing."""
        tracer = SessionTracer()
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        # Mock sync add_event
        tracer.add_event = Mock(side_effect=lambda e: events_captured.append(e))
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Sync response"))]
        mock_response.usage = mock_openai_response["usage"]
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create = Mock(return_value=mock_response)
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="test_sync_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            # Mock session context
            tracer.current_session = Mock(session_id="sync_session")
            tracer.in_timestep = True
            
            # Set context
            set_session_tracer(tracer)
            set_system_id("test_sync_agent")
            set_turn_number(1)
            
            response = lm.generate("Hello sync!")
        
        assert response.text == "Sync response"
        
        # Verify event was captured
        assert len(events_captured) >= 1
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.system_instance_id == "test_sync_agent"


class TestLMTracingContextPropagation:
    """Test context propagation through LM calls."""
    
    @pytest.mark.asyncio
    async def test_context_preserved_across_lm_calls(self, mock_openai_response):
        """Test that tracing context is preserved across multiple LM calls."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="multi_call_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("multi_call_session"):
                # Set initial context
                set_session_tracer(tracer)
                set_system_id("multi_call_agent")
                
                # First call with turn 1
                set_turn_number(1)
                response1 = await lm.agenerate("First prompt")
                
                # Second call with turn 2
                set_turn_number(2)
                response2 = await lm.agenerate("Second prompt")
                
                # Third call with turn 3
                set_turn_number(3)
                response3 = await lm.agenerate("Third prompt")
        
        # Verify all events were captured
        assert len(events_captured) == 3
        
        # Verify each event has correct turn number
        for i, event in enumerate(events_captured):
            assert isinstance(event, CAISEvent)
            assert event.system_instance_id == "multi_call_agent"
            assert event.time_record.message_time == i + 1
    
    @pytest.mark.asyncio
    async def test_nested_lm_calls_with_tracing(self, mock_openai_response):
        """Test nested LM calls maintain proper tracing context."""
        tracer = SessionTracer()
        events_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        tracer.add_event = capture_event
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm1 = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="outer_agent",
            enable_v2_tracing=True
        )
        
        lm2 = LM(
            provider="openai",
            model="gpt-3.5-turbo",
            session_tracer=tracer,
            system_id="inner_agent",
            enable_v2_tracing=True
        )
        
        async def inner_function():
            """Inner function that makes an LM call."""
            set_system_id("inner_agent")
            response = await lm2.agenerate("Inner prompt")
            return response.text
        
        # Patch both clients
        with patch.object(lm1, '_get_client', return_value=mock_client), \
             patch.object(lm2, '_get_client', return_value=mock_client):
            
            async with tracer.start_session("nested_session"):
                # Set context
                set_session_tracer(tracer)
                set_system_id("outer_agent")
                set_turn_number(1)
                
                # Outer LM call
                response1 = await lm1.agenerate("Outer prompt")
                
                # Call inner function (which makes another LM call)
                inner_result = await inner_function()
                
                # Another outer LM call
                set_system_id("outer_agent")
                response2 = await lm1.agenerate(f"Process: {inner_result}")
        
        # Verify all events were captured with correct system IDs
        assert len(events_captured) == 3
        assert events_captured[0].system_instance_id == "outer_agent"
        assert events_captured[1].system_instance_id == "inner_agent"
        assert events_captured[2].system_instance_id == "outer_agent"


class TestLMTracingIntegrationWithEnvironment:
    """Test LM tracing integration with environment interactions."""
    
    @pytest.mark.asyncio
    async def test_lm_with_environment_messages(self, mock_openai_response):
        """Test LM tracing alongside environment message tracking."""
        tracer = SessionTracer()
        events_captured = []
        messages_captured = []
        
        async def capture_event(event):
            events_captured.append(event)
        
        async def capture_message(message):
            messages_captured.append(message)
        
        tracer.add_event = capture_event
        original_add_message = tracer.add_message
        
        async def mock_add_message(message):
            await capture_message(message)
            await original_add_message(message)
        
        tracer.add_message = mock_add_message
        
        # Mock the OpenAI client
        mock_client = Mock()
        mock_create = AsyncMock(return_value=mock_openai_response)
        mock_client.chat.completions.create = mock_create
        
        lm = LM(
            provider="openai",
            model="gpt-4",
            session_tracer=tracer,
            system_id="env_agent",
            enable_v2_tracing=True
        )
        
        # Patch the client
        with patch.object(lm, '_get_client', return_value=mock_client):
            async with tracer.start_session("env_session"):
                set_session_tracer(tracer)
                
                # Simulate environment interaction
                async with tracer.timestep(1):
                    # Environment observation
                    env_message = SessionEventMessage(
                        content={
                            "origin_system_id": "test_env",
                            "payload": {"observation": "You see a door"}
                        },
                        message_type="observation",
                        time_record=TimeRecord(
                            event_time="2024-01-01T00:00:00",
                            message_time=1
                        )
                    )
                    await tracer.add_message(env_message)
                    
                    # Agent processes observation
                    set_system_id("env_agent")
                    set_turn_number(1)
                    response = await lm.agenerate(
                        f"Environment: {env_message.content['payload']['observation']}"
                    )
                    
                    # Agent action
                    action_message = SessionEventMessage(
                        content={
                            "origin_system_id": "env_agent",
                            "payload": {"action": "open_door"}
                        },
                        message_type="action",
                        time_record=TimeRecord(
                            event_time="2024-01-01T00:00:01",
                            message_time=1
                        )
                    )
                    await tracer.add_message(action_message)
        
        # Verify both messages and events were captured
        assert len(messages_captured) == 2
        assert len(events_captured) == 1
        
        # Verify message order
        assert messages_captured[0].message_type == "observation"
        assert messages_captured[1].message_type == "action"
        
        # Verify LM event
        assert events_captured[0].system_instance_id == "env_agent"
        assert events_captured[0].model_name == "gpt-4"