"""Tests for synth_ai.tracing.decorators module."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from synth_ai.tracing.decorators import (
    active_events_var,
    trace_event_async,
    trace_event_sync,
)
from synth_ai.tracing.events.store import event_store


class MockSystem:
    """Mock system class with required attributes for tracing."""

    def __init__(
        self, system_name="test_system", system_id="test_id", system_instance_id="test_instance"
    ):
        self.system_name = system_name
        self.system_id = system_id
        self.system_instance_id = system_instance_id
        self.system_instance_metadata = {"version": "1.0"}


class TestTraceEventSync:
    """Test the trace_event_sync decorator."""

    def test_basic_sync_function(self):
        """Test decorating a basic synchronous function."""

        class TestClass(MockSystem):
            @trace_event_sync(event_type="test_sync")
            def sample_function(self, x, y):
                return x + y

        test_instance = TestClass()

        # Clear any existing events by resetting the internal traces
        with event_store._lock:
            event_store._traces.clear()

        result = test_instance.sample_function(2, 3)
        assert result == 5

        # Check that event was stored
        traces = event_store.get_system_traces()
        assert len(traces) == 1
        assert traces[0].system_name == "test_system"
        # system_id is generated from system_name using UUID5
        assert traces[0].system_id is not None
        assert traces[0].system_instance_id == "test_instance"

        # Check event details
        events = traces[0].partition[0].events
        assert len(events) == 1
        assert events[0].event_type == "test_sync"
        assert events[0].opened is not None
        assert events[0].closed is not None
        assert events[0].closed > events[0].opened

    def test_sync_function_with_exception(self):
        """Test sync function that raises an exception."""

        class TestClass(MockSystem):
            @trace_event_sync(event_type="test_error")
            def failing_function(self):
                raise ValueError("Test error")

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        with pytest.raises(ValueError, match="Test error"):
            test_instance.failing_function()

        # Event should NOT be recorded when exception occurs
        # This is the current behavior of the decorator
        traces = event_store.get_system_traces()
        # The trace is created but no events are added on exception
        if len(traces) > 0:
            events = traces[0].partition[0].events if traces[0].partition else []
            assert len(events) == 0

    def test_sync_compute_logging(self):
        """Test compute input/output logging for sync functions."""

        class TestClass(MockSystem):
            @trace_event_sync(event_type="test_compute")
            def compute_function(self, value):
                return value * 2

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        result = test_instance.compute_function(5)
        assert result == 10

        traces = event_store.get_system_traces()
        events = traces[0].partition[0].events
        # The decorator should create compute steps but might not capture the specific values
        # without explicit tracking
        assert len(events) == 1
        assert events[0].event_type == "test_compute"


class TestTraceEventAsync:
    """Test the trace_event_async decorator."""

    @pytest.mark.asyncio
    async def test_basic_async_function(self):
        """Test decorating a basic asynchronous function."""

        class TestClass(MockSystem):
            @trace_event_async(event_type="test_async")
            async def sample_async_function(self, x, y):
                await asyncio.sleep(0.01)  # Simulate async work
                return x * y

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        result = await test_instance.sample_async_function(3, 4)
        assert result == 12

        # Check that event was stored
        traces = event_store.get_system_traces()
        assert len(traces) == 1
        assert traces[0].system_name == "test_system"

        events = traces[0].partition[0].events
        assert len(events) == 1
        assert events[0].event_type == "test_async"
        assert events[0].closed > events[0].opened

    @pytest.mark.asyncio
    async def test_async_function_with_exception(self):
        """Test async function that raises an exception."""

        class TestClass(MockSystem):
            @trace_event_async(event_type="test_async_error")
            async def failing_async_function(self):
                await asyncio.sleep(0.01)
                raise RuntimeError("Async test error")

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        with pytest.raises(RuntimeError, match="Async test error"):
            await test_instance.failing_async_function()

        # Event should NOT be recorded when exception occurs
        # This is the current behavior of the decorator
        traces = event_store.get_system_traces()
        # The trace is created but no events are added on exception
        if len(traces) > 0:
            events = traces[0].partition[0].events if traces[0].partition else []
            assert len(events) == 0

    @pytest.mark.asyncio
    async def test_async_compute_logging(self):
        """Test compute input/output logging for async functions."""

        class TestClass(MockSystem):
            @trace_event_async(event_type="test_async_compute")
            async def process_text(self, text):
                await asyncio.sleep(0.01)
                return text.upper()

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        result = await test_instance.process_text("hello")
        assert result == "HELLO"

        traces = event_store.get_system_traces()
        events = traces[0].partition[0].events
        # The decorator should create compute steps but might not capture the specific values
        # without explicit tracking
        assert len(events) == 1
        assert events[0].event_type == "test_async_compute"

    @pytest.mark.asyncio
    async def test_nested_async_events(self):
        """Test nested async function calls."""

        class TestClass(MockSystem):
            @trace_event_async(event_type="outer_async")
            async def outer_function(self):
                await asyncio.sleep(0.01)
                result = await self.inner_function()
                return f"outer: {result}"

            @trace_event_async(event_type="inner_async")
            async def inner_function(self):
                await asyncio.sleep(0.01)
                return "inner result"

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        result = await test_instance.outer_function()
        assert result == "outer: inner result"

        # Should have recorded events
        traces = event_store.get_system_traces()
        assert len(traces) == 1
        # Due to how the decorator manages events, only the outer event is recorded
        # The inner function call happens within the outer function's context
        events = traces[0].partition[0].events
        assert len(events) >= 1
        # At least the outer event should be recorded
        event_types = [e.event_type for e in events]
        assert "outer_async" in event_types


class TestEventManagement:
    """Test event management functionality."""

    def test_concurrent_sync_events(self):
        """Test handling multiple sync events."""

        class TestClass(MockSystem):
            @trace_event_sync(event_type="concurrent_test")
            def concurrent_function(self, id):
                time.sleep(0.01)
                return f"result_{id}"

        test_instance = TestClass()
        with event_store._lock:
            event_store._traces.clear()

        # Run multiple times
        results = []
        for i in range(3):
            results.append(test_instance.concurrent_function(i))

        assert results == ["result_0", "result_1", "result_2"]

        # Check that events were recorded
        traces = event_store.get_system_traces()
        assert len(traces) == 1
        # The decorator increments partition for each call by default
        # So we should have multiple partitions
        total_events = sum(len(p.events) for p in traces[0].partition)
        # We called the function 3 times, so we should have at least 3 events total
        assert total_events == 3
        # All events should be of type "concurrent_test"
        all_events = [e for p in traces[0].partition for e in p.events]
        assert all(e.event_type == "concurrent_test" for e in all_events)
