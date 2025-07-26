"""
End-to-end tests for v2 tracing system.

This module tests:
- Complete trace flow from agent to environment
- Trace serialization and deserialization
- Hook integration
- OpenTelemetry export
- Performance characteristics
"""

import pytest
import asyncio
import json
import uuid
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp

from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord, CAISEvent,
    RuntimeEvent, EnvironmentEvent, SessionTrace
)
from synth_ai.tracing_v2.decorators import (
    ai_call, environment_step, runtime_operation, function_call,
    set_session_tracer, set_system_id, set_turn_number,
    setup_otel_tracer
)
from synth_ai.tracing_v2.hooks import TraceHook, TraceStateHook
from synth_ai.lm.core.main_v2 import LM
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, InMemorySpanExporter


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.state = {
            "position": [0, 0],
            "health": 100,
            "inventory": {}
        }
        self.steps = 0
    
    @environment_step
    def step(self, action: str) -> Dict[str, Any]:
        """Take a step in the environment."""
        self.steps += 1
        
        # Update state based on action
        if action == "move_north":
            self.state["position"][1] += 1
        elif action == "move_south":
            self.state["position"][1] -= 1
        elif action == "move_east":
            self.state["position"][0] += 1
        elif action == "move_west":
            self.state["position"][0] -= 1
        elif action == "take_damage":
            self.state["health"] -= 10
        
        # Calculate reward
        reward = 1.0 if self.state["health"] > 0 else -10.0
        done = self.state["health"] <= 0 or self.steps >= 10
        
        return {
            "observation": f"You are at position {self.state['position']} with {self.state['health']} health",
            "reward": reward,
            "done": done,
            "info": {"steps": self.steps}
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        self.state = {
            "position": [0, 0],
            "health": 100,
            "inventory": {}
        }
        self.steps = 0
        return {
            "observation": f"You are at position {self.state['position']} with {self.state['health']} health"
        }


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, lm: Optional[LM] = None):
        self.agent_id = agent_id
        self.lm = lm
        self.action_history = []
    
    @ai_call
    async def think(self, observation: str) -> Dict[str, Any]:
        """Process observation and decide on action."""
        # Simulate LM call if available
        if self.lm:
            prompt = f"Observation: {observation}\nWhat action should I take?"
            response = await self.lm.agenerate(prompt)
            action = self._parse_action(response.text)
        else:
            # Simple rule-based action
            if "health" in observation and "100" in observation:
                action = "move_north"
            elif "health" in observation and int(observation.split()[-2]) < 50:
                action = "move_south"
            else:
                action = "move_east"
        
        self.action_history.append(action)
        
        return {
            "response": f"I will {action}",
            "action": action,
            "model": "mock-model",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
    
    def _parse_action(self, response: str) -> str:
        """Parse action from LM response."""
        actions = ["move_north", "move_south", "move_east", "move_west"]
        for action in actions:
            if action in response.lower():
                return action
        return "move_north"  # default


class TestEndToEndTracing:
    """Test complete tracing flow."""
    
    @pytest.fixture
    def setup_otel_exporter(self):
        """Set up OpenTelemetry with in-memory exporter for testing."""
        exporter = InMemorySpanExporter()
        setup_otel_tracer()
        
        # Add exporter to tracer provider
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        yield exporter
        
        # Clean up
        trace._TRACER_PROVIDER = None
    
    @pytest.mark.asyncio
    async def test_agent_environment_loop(self, setup_otel_exporter):
        """Test complete agent-environment interaction loop with tracing."""
        tracer = SessionTracer()
        env = MockEnvironment("test_env")
        agent = MockAgent("test_agent")
        
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id, {"experiment": "e2e_test"}):
            # Set up context
            set_session_tracer(tracer)
            
            # Reset environment
            obs_data = env.reset()
            
            # Run interaction loop
            for turn in range(5):
                async with tracer.timestep(turn):
                    # Record environment observation
                    obs_message = SessionEventMessage(
                        content={
                            "origin_system_id": env.env_id,
                            "payload": obs_data
                        },
                        message_type="observation",
                        time_record=TimeRecord(
                            event_time=time.time(),
                            message_time=turn
                        )
                    )
                    await tracer.add_message(obs_message)
                    
                    # Agent thinks
                    set_system_id(agent.agent_id)
                    set_turn_number(turn)
                    thought = await agent.think(obs_data["observation"])
                    
                    # Record agent action
                    action_message = SessionEventMessage(
                        content={
                            "origin_system_id": agent.agent_id,
                            "payload": {"action": thought["action"]}
                        },
                        message_type="action",
                        time_record=TimeRecord(
                            event_time=time.time(),
                            message_time=turn
                        )
                    )
                    await tracer.add_message(action_message)
                    
                    # Environment step
                    set_system_id(env.env_id)
                    step_result = env.step(thought["action"])
                    obs_data = {"observation": step_result["observation"]}
                    
                    if step_result["done"]:
                        break
        
        # Get the complete trace
        trace_data = tracer.get_session_trace(session_id)
        
        # Verify trace structure
        assert trace_data is not None
        assert trace_data.session_id == session_id
        assert trace_data.metadata["experiment"] == "e2e_test"
        assert len(trace_data.timesteps) <= 5
        
        # Verify messages in each timestep
        for timestep in trace_data.timesteps:
            messages = timestep.messages
            assert len(messages) >= 2  # At least observation and action
            
            # Check message types
            message_types = [msg.message_type for msg in messages]
            assert "observation" in message_types
            assert "action" in message_types
        
        # Verify events
        assert len(trace_data.event_history) > 0
        
        # Check AI events
        ai_events = [e for e in trace_data.event_history if isinstance(e, CAISEvent)]
        assert len(ai_events) == len(trace_data.timesteps)
        
        # Check environment events
        env_events = [e for e in trace_data.event_history if isinstance(e, EnvironmentEvent)]
        assert len(env_events) == len(trace_data.timesteps)
        
        # Verify OpenTelemetry spans were created
        setup_otel_exporter.shutdown()
        spans = setup_otel_exporter.get_finished_spans()
        assert len(spans) > 0
    
    @pytest.mark.asyncio
    async def test_trace_serialization_deserialization(self, tmp_path):
        """Test saving and loading traces."""
        tracer = SessionTracer()
        env = MockEnvironment("serialize_env")
        agent = MockAgent("serialize_agent")
        
        session_id = str(uuid.uuid4())
        
        # Create a trace
        async with tracer.start_session(session_id, {"test": "serialization"}):
            set_session_tracer(tracer)
            
            obs_data = env.reset()
            
            async with tracer.timestep(1):
                # Add messages
                await tracer.add_message(SessionEventMessage(
                    content={"origin_system_id": env.env_id, "payload": obs_data},
                    message_type="observation"
                ))
                
                set_system_id(agent.agent_id)
                set_turn_number(1)
                thought = await agent.think(obs_data["observation"])
                
                await tracer.add_message(SessionEventMessage(
                    content={"origin_system_id": agent.agent_id, "payload": {"action": thought["action"]}},
                    message_type="action"
                ))
        
        # Save trace
        trace_file = tmp_path / f"{session_id}.json"
        tracer.save_trace(session_id, str(trace_file))
        
        assert trace_file.exists()
        
        # Load and verify
        with open(trace_file) as f:
            loaded_data = json.load(f)
        
        assert loaded_data["session_id"] == session_id
        assert loaded_data["metadata"]["test"] == "serialization"
        assert len(loaded_data["session_time_steps"]) == 1
        assert len(loaded_data["session_time_steps"][0]["step_messages"]) == 2
        assert len(loaded_data["event_history"]) >= 1
        
        # Verify data integrity
        timestep = loaded_data["session_time_steps"][0]
        assert timestep["step_id"] == 1
        
        # Check messages
        messages = timestep["step_messages"]
        assert messages[0]["message_type"] == "observation"
        assert messages[1]["message_type"] == "action"
        
        # Check events
        events = loaded_data["event_history"]
        assert any(e.get("model_name") == "mock-model" for e in events)
    
    @pytest.mark.asyncio
    async def test_trace_hooks_integration(self):
        """Test trace hooks in end-to-end scenario."""
        hook_calls = {
            "events": [],
            "messages": [],
            "state_changes": []
        }
        
        class TestHook(TraceHook):
            async def on_event(self, event, session_id):
                hook_calls["events"].append({
                    "type": type(event).__name__,
                    "session_id": session_id
                })
            
            async def on_message(self, message, session_id):
                hook_calls["messages"].append({
                    "type": message.message_type,
                    "session_id": session_id
                })
        
        class TestStateHook(TraceStateHook):
            async def on_state_change(self, tracer, state_type, **kwargs):
                hook_calls["state_changes"].append({
                    "state_type": state_type,
                    "kwargs": kwargs
                })
        
        tracer = SessionTracer()
        tracer.add_hook(TestHook())
        tracer.add_hook(TestStateHook())
        
        env = MockEnvironment("hook_env")
        agent = MockAgent("hook_agent")
        
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id):
            set_session_tracer(tracer)
            
            obs_data = env.reset()
            
            async with tracer.timestep(1):
                await tracer.add_message(SessionEventMessage(
                    content={"origin_system_id": env.env_id, "payload": obs_data},
                    message_type="observation"
                ))
                
                set_system_id(agent.agent_id)
                set_turn_number(1)
                await agent.think(obs_data["observation"])
        
        # Give async operations time to complete
        await asyncio.sleep(0.1)
        
        # Verify hooks were called
        assert len(hook_calls["events"]) >= 1
        assert len(hook_calls["messages"]) >= 1
        assert len(hook_calls["state_changes"]) >= 2  # session_start and timestep_start
        
        # Verify event types
        event_types = [e["type"] for e in hook_calls["events"]]
        assert "CAISEvent" in event_types
        
        # Verify message types
        message_types = [m["type"] for m in hook_calls["messages"]]
        assert "observation" in message_types
        
        # Verify state changes
        state_types = [s["state_type"] for s in hook_calls["state_changes"]]
        assert "session_start" in state_types
        assert "timestep_start" in state_types
    
    @pytest.mark.asyncio
    async def test_lm_integration_e2e(self):
        """Test end-to-end with real LM integration."""
        # Mock LM responses
        mock_responses = [
            {"text": "I should move north to explore", "usage": {"prompt_tokens": 30, "completion_tokens": 10}},
            {"text": "I'll go east now", "usage": {"prompt_tokens": 35, "completion_tokens": 8}},
            {"text": "Moving south seems wise", "usage": {"prompt_tokens": 40, "completion_tokens": 9}}
        ]
        
        response_iter = iter(mock_responses)
        
        # Create mock LM
        lm = LM(
            provider="openai",
            model="gpt-4",
            enable_v2_tracing=True
        )
        
        # Mock the generate method
        async def mock_generate(prompt, **kwargs):
            response_data = next(response_iter)
            response = Mock()
            response.text = response_data["text"]
            response.usage = response_data["usage"]
            return response
        
        lm.agenerate = mock_generate
        
        tracer = SessionTracer()
        lm._session_tracer = tracer
        lm._system_id = "lm_agent"
        
        env = MockEnvironment("lm_env")
        agent = MockAgent("lm_agent", lm=lm)
        
        session_id = str(uuid.uuid4())
        
        async with tracer.start_session(session_id, {"with_lm": True}):
            set_session_tracer(tracer)
            
            obs_data = env.reset()
            
            for turn in range(3):
                async with tracer.timestep(turn):
                    # Environment observation
                    await tracer.add_message(SessionEventMessage(
                        content={"origin_system_id": env.env_id, "payload": obs_data},
                        message_type="observation",
                        time_record=TimeRecord(event_time=time.time(), message_time=turn)
                    ))
                    
                    # Agent thinks (with LM)
                    set_system_id(agent.agent_id)
                    set_turn_number(turn)
                    thought = await agent.think(obs_data["observation"])
                    
                    # Agent action
                    await tracer.add_message(SessionEventMessage(
                        content={"origin_system_id": agent.agent_id, "payload": {"action": thought["action"]}},
                        message_type="action",
                        time_record=TimeRecord(event_time=time.time(), message_time=turn)
                    ))
                    
                    # Environment responds
                    set_system_id(env.env_id)
                    step_result = env.step(thought["action"])
                    obs_data = {"observation": step_result["observation"]}
        
        # Verify trace
        trace_data = tracer.get_session_trace(session_id)
        assert len(trace_data.timesteps) == 3
        assert len(trace_data.event_history) >= 3  # At least one AI event per turn
        
        # Verify agent took different actions based on LM
        assert agent.action_history == ["move_north", "move_east", "move_south"]
    
    @pytest.mark.asyncio
    async def test_performance_characteristics(self):
        """Test that tracing overhead stays within bounds."""
        import statistics
        
        # Run without tracing
        env = MockEnvironment("perf_env")
        agent = MockAgent("perf_agent")
        
        no_trace_times = []
        for _ in range(100):
            start = time.perf_counter()
            
            obs_data = env.reset()
            for _ in range(10):
                thought = await agent.think(obs_data["observation"])
                step_result = env.step(thought["action"])
                obs_data = {"observation": step_result["observation"]}
                if step_result["done"]:
                    break
            
            no_trace_times.append(time.perf_counter() - start)
        
        # Run with tracing
        tracer = SessionTracer()
        trace_times = []
        
        for i in range(100):
            start = time.perf_counter()
            
            async with tracer.start_session(f"perf_session_{i}"):
                set_session_tracer(tracer)
                
                obs_data = env.reset()
                for turn in range(10):
                    async with tracer.timestep(turn):
                        await tracer.add_message(SessionEventMessage(
                            content={"origin_system_id": env.env_id, "payload": obs_data},
                            message_type="observation"
                        ))
                        
                        set_system_id(agent.agent_id)
                        set_turn_number(turn)
                        thought = await agent.think(obs_data["observation"])
                        
                        await tracer.add_message(SessionEventMessage(
                            content={"origin_system_id": agent.agent_id, "payload": {"action": thought["action"]}},
                            message_type="action"
                        ))
                        
                        set_system_id(env.env_id)
                        step_result = env.step(thought["action"])
                        obs_data = {"observation": step_result["observation"]}
                        
                        if step_result["done"]:
                            break
            
            trace_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        no_trace_mean = statistics.mean(no_trace_times)
        trace_mean = statistics.mean(trace_times)
        
        overhead_percent = ((trace_mean - no_trace_mean) / no_trace_mean) * 100
        
        # Assert overhead is less than 5%
        assert overhead_percent < 5.0, f"Tracing overhead is {overhead_percent:.2f}%, expected < 5%"
        
        # Also check that traces were actually created
        assert len(tracer._sessions) == 100
    
    @pytest.mark.asyncio
    async def test_error_recovery_e2e(self):
        """Test that tracing handles errors gracefully in e2e scenario."""
        tracer = SessionTracer()
        env = MockEnvironment("error_env")
        
        # Agent that occasionally fails
        class ErrorAgent(MockAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            @ai_call
            async def think(self, observation: str) -> Dict[str, Any]:
                self.call_count += 1
                if self.call_count == 2:
                    raise ValueError("Simulated agent error")
                return await super().think(observation)
        
        agent = ErrorAgent("error_agent")
        session_id = str(uuid.uuid4())
        
        errors_encountered = []
        
        async with tracer.start_session(session_id):
            set_session_tracer(tracer)
            
            obs_data = env.reset()
            
            for turn in range(4):
                try:
                    async with tracer.timestep(turn):
                        await tracer.add_message(SessionEventMessage(
                            content={"origin_system_id": env.env_id, "payload": obs_data},
                            message_type="observation"
                        ))
                        
                        set_system_id(agent.agent_id)
                        set_turn_number(turn)
                        thought = await agent.think(obs_data["observation"])
                        
                        await tracer.add_message(SessionEventMessage(
                            content={"origin_system_id": agent.agent_id, "payload": {"action": thought["action"]}},
                            message_type="action"
                        ))
                        
                        set_system_id(env.env_id)
                        step_result = env.step(thought["action"])
                        obs_data = {"observation": step_result["observation"]}
                
                except ValueError as e:
                    errors_encountered.append(str(e))
                    # Continue with default action
                    await tracer.add_message(SessionEventMessage(
                        content={"origin_system_id": agent.agent_id, "payload": {"action": "move_north", "error": str(e)}},
                        message_type="action"
                    ))
                    
                    step_result = env.step("move_north")
                    obs_data = {"observation": step_result["observation"]}
        
        # Verify trace despite errors
        trace_data = tracer.get_session_trace(session_id)
        assert trace_data is not None
        assert len(trace_data.timesteps) == 4
        assert len(errors_encountered) == 1
        assert "Simulated agent error" in errors_encountered[0]
        
        # Check that error was recorded in messages
        error_messages = []
        for timestep in trace_data.timesteps:
            for msg in timestep.messages:
                if "error" in msg.content.get("payload", {}):
                    error_messages.append(msg)
        
        assert len(error_messages) == 1
        assert error_messages[0].content["payload"]["error"] == "Simulated agent error"