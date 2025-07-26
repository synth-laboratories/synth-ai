"""
Tests for v2 decorator functionality and parity with manual tracing.

This module tests:
- Decorator functionality from decorators_v3_improved.py
- Parity between decorator-based and manual tracing
- Context propagation across async/sync boundaries
- Performance overhead validation
"""

import pytest
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import contextvars

from synth_ai.tracing_v2.decorators import (
    ai_call, environment_step, runtime_operation, function_call,
    set_session_tracer, get_session_tracer, set_system_id, set_turn_number,
    _session_tracer_ctx, _system_id_ctx, _turn_number_ctx,
    setup_otel_tracer, mask_pii, calculate_cost
)
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord, CAISEvent,
    RuntimeEvent, EnvironmentEvent
)
from synth_ai.tracing_v2.config import get_config, TracingConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


class TestDecoratorBasics:
    """Test basic decorator functionality."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            otel_service_name="test-service",
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.fixture
    def tracer(self):
        """Create a SessionTracer for testing."""
        return SessionTracer()
    
    @pytest.fixture
    def setup_otel(self):
        """Set up OpenTelemetry for testing."""
        setup_otel_tracer()
        yield
        # Clean up
        trace._TRACER_PROVIDER = None
    
    def test_ai_call_decorator_sync(self, tracer, mock_config, setup_otel):
        """Test @ai_call decorator on synchronous function."""
        set_session_tracer(tracer)
        set_system_id("test_agent")
        set_turn_number(1)
        
        @ai_call
        def test_llm_call(prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
            return {
                "response": f"Response to: {prompt}",
                "model": model,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        
        # Call the decorated function
        result = test_llm_call("Hello, world!")
        
        assert result["response"] == "Response to: Hello, world!"
        assert result["model"] == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_ai_call_decorator_async(self, tracer, mock_config, setup_otel):
        """Test @ai_call decorator on asynchronous function."""
        set_session_tracer(tracer)
        set_system_id("test_agent")
        set_turn_number(1)
        
        events_captured = []
        
        # Mock the tracer's add_event method to capture events
        original_add_event = tracer.add_event
        async def mock_add_event(event):
            events_captured.append(event)
            await original_add_event(event)
        
        tracer.add_event = mock_add_event
        
        @ai_call
        async def test_async_llm_call(prompt: str, model: str = "claude-3") -> Dict[str, Any]:
            await asyncio.sleep(0.01)  # Simulate async operation
            return {
                "response": f"Async response to: {prompt}",
                "model": model,
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40
                }
            }
        
        # Start a session to capture events
        async with tracer.start_session("test_session"):
            result = await test_async_llm_call("Test prompt")
        
        assert result["response"] == "Async response to: Test prompt"
        assert result["model"] == "claude-3"
        
        # Verify event was captured
        assert len(events_captured) > 0
        event = events_captured[0]
        assert isinstance(event, CAISEvent)
        assert event.model_name == "claude-3"
        assert event.prompt_tokens == 15
        assert event.completion_tokens == 25
        assert event.total_tokens == 40
    
    def test_environment_step_decorator(self, tracer, mock_config, setup_otel):
        """Test @environment_step decorator."""
        set_session_tracer(tracer)
        set_system_id("test_env")
        set_turn_number(1)
        
        @environment_step
        def take_action(action: str, state: Dict[str, Any]) -> Dict[str, Any]:
            new_state = state.copy()
            new_state["last_action"] = action
            new_state["step"] = state.get("step", 0) + 1
            return {
                "state": new_state,
                "reward": 1.0 if action == "good" else -1.0,
                "done": new_state["step"] >= 10
            }
        
        initial_state = {"health": 100, "position": [0, 0]}
        result = take_action("good", initial_state)
        
        assert result["state"]["last_action"] == "good"
        assert result["state"]["step"] == 1
        assert result["reward"] == 1.0
        assert result["done"] is False
    
    def test_runtime_operation_decorator(self, tracer, mock_config, setup_otel):
        """Test @runtime_operation decorator."""
        set_session_tracer(tracer)
        set_system_id("test_runtime")
        
        @runtime_operation(operation_type="data_processing")
        def process_data(data: List[int]) -> int:
            return sum(data)
        
        result = process_data([1, 2, 3, 4, 5])
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_function_call_decorator_async(self, tracer, mock_config, setup_otel):
        """Test @function_call decorator on async function."""
        set_session_tracer(tracer)
        
        @function_call
        async def async_compute(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x + y
        
        result = await async_compute(10, 20)
        assert result == 30


class TestDecoratorContextPropagation:
    """Test context propagation across boundaries."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.fixture
    def setup_otel(self):
        """Set up OpenTelemetry for testing."""
        setup_otel_tracer()
        yield
        trace._TRACER_PROVIDER = None
    
    @pytest.mark.asyncio
    async def test_context_propagation_async_to_sync(self, mock_config, setup_otel):
        """Test context propagation from async to sync functions."""
        tracer = SessionTracer()
        
        @function_call
        def sync_helper(value: int) -> int:
            # Context should be preserved
            assert get_session_tracer() == tracer
            return value * 2
        
        @ai_call
        async def async_main(prompt: str) -> Dict[str, Any]:
            # Set context in async function
            set_session_tracer(tracer)
            
            # Call sync function
            doubled = sync_helper(5)
            
            return {
                "response": f"Doubled: {doubled}",
                "model": "test-model"
            }
        
        async with tracer.start_session("test_session"):
            result = await async_main("Test")
            assert result["response"] == "Doubled: 10"
    
    def test_context_propagation_thread_pool(self, mock_config, setup_otel):
        """Test context propagation in thread pool."""
        tracer = SessionTracer()
        set_session_tracer(tracer)
        set_system_id("thread_test")
        
        results = []
        
        @function_call
        def thread_worker(task_id: int) -> Dict[str, Any]:
            # Context should be preserved in thread
            current_tracer = get_session_tracer()
            return {
                "task_id": task_id,
                "tracer_preserved": current_tracer == tracer
            }
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(thread_worker, i)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                results.append(result)
        
        # Verify all threads preserved context
        assert all(r["tracer_preserved"] for r in results)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_nested_decorator_context(self, mock_config, setup_otel):
        """Test context preservation with nested decorators."""
        tracer = SessionTracer()
        events = []
        
        async def capture_event(event):
            events.append(event)
        
        tracer.add_event = capture_event
        
        @runtime_operation(operation_type="helper")
        def helper_function(x: int) -> int:
            return x * 2
        
        @ai_call
        async def outer_function(prompt: str) -> Dict[str, Any]:
            set_session_tracer(tracer)
            set_system_id("nested_test")
            
            # Call decorated helper
            value = helper_function(10)
            
            return {
                "response": f"Result: {value}",
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10}
            }
        
        async with tracer.start_session("nested_session"):
            result = await outer_function("Test")
        
        assert result["response"] == "Result: 20"
        # Should have events from both decorators
        assert len(events) >= 1  # At least the AI call event


class TestDecoratorParityWithManual:
    """Test that decorators produce same results as manual tracing."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.fixture
    def setup_otel(self):
        """Set up OpenTelemetry for testing."""
        setup_otel_tracer()
        yield
        trace._TRACER_PROVIDER = None
    
    @pytest.mark.asyncio
    async def test_ai_call_parity(self, mock_config, setup_otel):
        """Test that @ai_call produces same events as manual tracing."""
        # Manual tracing
        manual_tracer = SessionTracer()
        manual_events = []
        
        async def manual_capture(event):
            manual_events.append(event)
        
        manual_tracer.add_event = manual_capture
        
        async with manual_tracer.start_session("manual_session"):
            # Manually create and add event
            event = CAISEvent(
                system_instance_id="test_agent",
                model_name="gpt-4",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                time_record=TimeRecord(
                    event_time=time.time(),
                    message_time=1
                )
            )
            await manual_tracer.add_event(event)
        
        # Decorator tracing
        decorator_tracer = SessionTracer()
        decorator_events = []
        
        async def decorator_capture(event):
            decorator_events.append(event)
        
        decorator_tracer.add_event = decorator_capture
        
        @ai_call
        async def test_function(prompt: str) -> Dict[str, Any]:
            return {
                "response": "Test response",
                "model": "gpt-4",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        
        set_session_tracer(decorator_tracer)
        set_system_id("test_agent")
        set_turn_number(1)
        
        async with decorator_tracer.start_session("decorator_session"):
            await test_function("Test prompt")
        
        # Compare events
        assert len(manual_events) == 1
        assert len(decorator_events) == 1
        
        manual_event = manual_events[0]
        decorator_event = decorator_events[0]
        
        # Compare key fields
        assert manual_event.system_instance_id == decorator_event.system_instance_id
        assert manual_event.model_name == decorator_event.model_name
        assert manual_event.prompt_tokens == decorator_event.prompt_tokens
        assert manual_event.completion_tokens == decorator_event.completion_tokens
        assert manual_event.total_tokens == decorator_event.total_tokens


class TestDecoratorPerformance:
    """Test decorator performance overhead."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=False,  # Disable OTel for pure overhead test
            emit_events=False  # Disable events for pure overhead test
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    def test_decorator_overhead_sync(self, mock_config):
        """Test that sync decorator overhead is < 5%."""
        # Undecorated function
        def undecorated_function(x: int, y: int) -> int:
            return x + y
        
        # Decorated function
        @function_call
        def decorated_function(x: int, y: int) -> int:
            return x + y
        
        # Measure undecorated performance
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            undecorated_function(10, 20)
        undecorated_time = time.perf_counter() - start
        
        # Measure decorated performance
        start = time.perf_counter()
        for _ in range(iterations):
            decorated_function(10, 20)
        decorated_time = time.perf_counter() - start
        
        # Calculate overhead
        overhead = (decorated_time - undecorated_time) / undecorated_time * 100
        
        # Assert overhead is less than 5%
        assert overhead < 5.0, f"Decorator overhead is {overhead:.2f}%, expected < 5%"
    
    @pytest.mark.asyncio
    async def test_decorator_overhead_async(self, mock_config):
        """Test that async decorator overhead is < 5%."""
        # Undecorated function
        async def undecorated_function(x: int, y: int) -> int:
            return x + y
        
        # Decorated function
        @function_call
        async def decorated_function(x: int, y: int) -> int:
            return x + y
        
        # Measure undecorated performance
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            await undecorated_function(10, 20)
        undecorated_time = time.perf_counter() - start
        
        # Measure decorated performance
        start = time.perf_counter()
        for _ in range(iterations):
            await decorated_function(10, 20)
        decorated_time = time.perf_counter() - start
        
        # Calculate overhead
        overhead = (decorated_time - undecorated_time) / undecorated_time * 100
        
        # Assert overhead is less than 5%
        assert overhead < 5.0, f"Decorator overhead is {overhead:.2f}%, expected < 5%"


class TestDecoratorUtilities:
    """Test decorator utility functions."""
    
    def test_mask_pii(self):
        """Test PII masking functionality."""
        text = "My email is john.doe@example.com and my phone is 555-123-4567"
        masked = mask_pii(text)
        
        assert "[EMAIL]" in masked
        assert "[PHONE]" in masked
        assert "john.doe@example.com" not in masked
        assert "555-123-4567" not in masked
    
    def test_calculate_cost(self):
        """Test cost calculation for known models."""
        # Test GPT-4
        cost = calculate_cost("gpt-4", 100, 200)
        expected = (100 * 30.0 + 200 * 60.0) / 1_000_000
        assert cost == expected
        
        # Test Claude-3 Opus
        cost = calculate_cost("claude-3-opus", 1000, 500)
        expected = (1000 * 15.0 + 500 * 75.0) / 1_000_000
        assert cost == expected
        
        # Test unknown model
        cost = calculate_cost("unknown-model", 100, 100)
        assert cost == 0.0
    
    def test_context_variable_management(self):
        """Test context variable getters and setters."""
        tracer = SessionTracer()
        
        # Test session tracer
        set_session_tracer(tracer)
        assert get_session_tracer() == tracer
        
        # Test system ID
        set_system_id("test_system")
        assert _system_id_ctx.get() == "test_system"
        
        # Test turn number
        set_turn_number(42)
        assert _turn_number_ctx.get() == 42
        
        # Clear and verify
        set_session_tracer(None)
        assert get_session_tracer() is None


class TestDecoratorErrorHandling:
    """Test decorator error handling."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.fixture
    def setup_otel(self):
        """Set up OpenTelemetry for testing."""
        setup_otel_tracer()
        yield
        trace._TRACER_PROVIDER = None
    
    def test_decorator_with_exception(self, mock_config, setup_otel):
        """Test decorator behavior when function raises exception."""
        @function_call
        def failing_function(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 2
        
        # Should work normally
        assert failing_function(5) == 10
        
        # Should propagate exception
        with pytest.raises(ValueError, match="Negative value not allowed"):
            failing_function(-5)
    
    @pytest.mark.asyncio
    async def test_async_decorator_with_exception(self, mock_config, setup_otel):
        """Test async decorator behavior with exceptions."""
        @ai_call
        async def failing_async_function(prompt: str) -> Dict[str, Any]:
            if "error" in prompt:
                raise RuntimeError("Simulated error")
            return {"response": "Success", "model": "test"}
        
        # Should work normally
        result = await failing_async_function("normal prompt")
        assert result["response"] == "Success"
        
        # Should propagate exception
        with pytest.raises(RuntimeError, match="Simulated error"):
            await failing_async_function("error prompt")
    
    def test_decorator_with_disabled_tracing(self, monkeypatch):
        """Test decorators work when tracing is disabled."""
        config = TracingConfig(enabled=False)
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        
        @ai_call
        def test_function(prompt: str) -> Dict[str, Any]:
            return {"response": "Test", "model": "test-model"}
        
        # Should work normally even with tracing disabled
        result = test_function("Test prompt")
        assert result["response"] == "Test"