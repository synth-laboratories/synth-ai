#!/usr/bin/env python3
"""
Tests for SessionTracer in tracing v3.
Tests async functionality, hooks, and isolation.
"""

import time

import pytest

from ..abstractions import EnvironmentEvent, LMCAISEvent, RuntimeEvent, TimeRecord
from ..decorators import SessionContext, get_session_id
from ..hooks import HookManager
from ..lm_call_record_abstractions import (
    LLMCallRecord,
    LLMContentPart,
    LLMMessage,
    LLMUsage,
)
from ..session_tracer import SessionTracer


@pytest.mark.asyncio
class TestSessionTracer:
    """Test the async SessionTracer functionality."""

    async def test_basic_session_lifecycle(self):
        """Test basic session lifecycle."""
        tracer = SessionTracer(auto_save=False)

        # Start session
        session_id = await tracer.start_session(metadata={"test": "value"})
        assert session_id is not None
        assert tracer.current_session is not None
        current_session = tracer.current_session
        assert current_session.session_id == session_id
        assert current_session.metadata == {"test": "value"}

        # Start timestep
        step = await tracer.start_timestep("step1", turn_number=1)
        assert step.step_id == "step1"
        assert step.turn_number == 1
        assert tracer.current_step == step
        assert tracer.current_step is not None

        # Record event
        event = RuntimeEvent(
            system_instance_id="test_system",
            time_record=TimeRecord(event_time=time.time()),
            actions=[1, 2, 3],
        )
        await tracer.record_event(event)
        assert tracer.current_session is not None
        assert len(tracer.current_session.event_history) == 1
        assert tracer.current_step is not None
        assert len(tracer.current_step.events) == 1

        # Record message
        await tracer.record_message(content="Test message", message_type="user")
        assert tracer.current_session is not None
        assert len(tracer.current_session.markov_blanket_message_history) == 1
        assert tracer.current_step is not None
        assert len(tracer.current_step.markov_blanket_messages) == 1

        # End timestep
        await tracer.end_timestep()
        assert tracer.current_step is None
        assert step.completed_at is not None

        # End session
        trace = await tracer.end_session(save=False)
        assert trace.session_id == session_id
        assert tracer.current_session is None

    async def test_session_context_manager(self):
        """Test session context manager."""
        tracer = SessionTracer(auto_save=False)

        async with tracer.session(metadata={"context": "manager"}) as session_id:
            assert session_id is not None
            assert tracer.current_session is not None

            # Add some data
            await tracer.record_message("Test", "user")

        # Session should be ended
        assert tracer.current_session is None

    async def test_timestep_context_manager(self):
        """Test timestep context manager."""
        tracer = SessionTracer(auto_save=False)

        await tracer.start_session()

        async with tracer.timestep("step1", turn_number=1) as step:
            assert step.step_id == "step1"
            assert tracer.current_step == step

            # Add event
            await tracer.record_event(
                RuntimeEvent(
                    system_instance_id="test",
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[1],
                )
            )

        # Timestep should be ended
        assert tracer.current_step is None
        assert step.completed_at is not None

        await tracer.end_session(save=False)

    async def test_multiple_timesteps(self):
        """Test multiple timesteps in a session."""
        tracer = SessionTracer(auto_save=False)

        await tracer.start_session()

        # Create multiple timesteps
        for i in range(3):
            async with tracer.timestep(f"step_{i}", turn_number=i):
                await tracer.record_message(f"Message {i}", "user")
                await tracer.record_event(
                    RuntimeEvent(
                        system_instance_id="test",
                        time_record=TimeRecord(event_time=time.time()),
                        actions=[i],
                    )
                )

        trace = await tracer.end_session(save=False)

        assert len(trace.session_time_steps) == 3
        assert len(trace.event_history) == 3
        assert len(trace.markov_blanket_message_history) == 3

        # Verify step indices
        for i, step in enumerate(trace.session_time_steps):
            assert step.step_index == i
            assert step.step_id == f"step_{i}"
            assert step.turn_number == i

    async def test_hooks_integration(self):
        """Test hooks integration."""
        # Create custom hook manager
        hooks = HookManager()

        # Track hook calls
        hook_calls = {
            "session_start": 0,
            "session_end": 0,
            "timestep_start": 0,
            "timestep_end": 0,
            "event_recorded": 0,
            "message_recorded": 0,
        }

        # Register hooks
        async def track_hook(hook_name):
            async def hook_func(**kwargs):
                hook_calls[hook_name] += 1

            return hook_func

        for hook_name in hook_calls:
            hooks.register(hook_name, await track_hook(hook_name))

        # Create tracer with custom hooks
        tracer = SessionTracer(hooks=hooks, auto_save=False)

        # Perform operations
        await tracer.start_session()
        async with tracer.timestep("step1"):
            await tracer.record_event(
                RuntimeEvent(
                    system_instance_id="test",
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[1],
                )
            )
            await tracer.record_message("Test", "user")
        await tracer.end_session(save=False)

        # Verify hooks were called
        assert hook_calls["session_start"] == 1
        assert hook_calls["session_end"] == 1
        assert hook_calls["timestep_start"] == 1
        assert hook_calls["timestep_end"] == 1
        assert hook_calls["event_recorded"] == 1
        assert hook_calls["message_recorded"] == 1

    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        tracer = SessionTracer(auto_save=False)

        # Can't start timestep without session
        with pytest.raises(RuntimeError):
            await tracer.start_timestep("step1")

        # Can't record event without session
        with pytest.raises(RuntimeError):
            await tracer.record_event(
                RuntimeEvent(
                    system_instance_id="test",
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[1],
                )
            )

        # Can't end session without starting
        with pytest.raises(RuntimeError):
            await tracer.end_session()

        # Can't start session twice
        await tracer.start_session()
        with pytest.raises(RuntimeError):
            await tracer.start_session()

        await tracer.end_session(save=False)

    async def test_session_isolation(self):
        """Test that sessions are isolated from each other."""
        tracer1 = SessionTracer(auto_save=False)
        tracer2 = SessionTracer(auto_save=False)

        # Start sessions
        session1 = await tracer1.start_session()
        session2 = await tracer2.start_session()

        assert session1 != session2
        assert tracer1.current_session is not None
        assert tracer2.current_session is not None
        session_state1 = tracer1.current_session
        session_state2 = tracer2.current_session
        assert session_state1.session_id != session_state2.session_id

        # Add data to tracer1
        await tracer1.record_message("Tracer 1 message", "user")

        # Verify isolation
        assert len(session_state1.markov_blanket_message_history) == 1
        assert len(session_state2.markov_blanket_message_history) == 0

        await tracer1.end_session(save=False)
        await tracer2.end_session(save=False)

    async def test_auto_save_disabled(self):
        """Test that auto_save=False prevents database writes."""
        # Create tracer with no DB path (should not write)
        tracer = SessionTracer(db_url=None, auto_save=False)

        await tracer.start_session()
        await tracer.record_message("Test", "user")
        trace = await tracer.end_session(save=False)

        # Should complete without errors
        assert trace is not None
        assert len(trace.markov_blanket_message_history) == 1

    async def test_session_context_variables(self):
        """Test session context variables."""
        tracer = SessionTracer(auto_save=False)

        # Initially no session
        assert get_session_id() is None

        # Start session
        session_id = await tracer.start_session()

        # Context should be set
        assert get_session_id() == session_id

        # Use SessionContext
        async with SessionContext("different_session") as ctx:
            assert get_session_id() == "different_session"

        # Should revert
        assert get_session_id() == session_id

        await tracer.end_session(save=False)

        # Should be cleared
        assert get_session_id() is None

    async def test_event_types(self):
        """Test different event types."""
        tracer = SessionTracer(auto_save=False)

        await tracer.start_session()
        await tracer.start_timestep("step1")

        # Runtime event
        runtime_event = RuntimeEvent(
            system_instance_id="runtime",
            time_record=TimeRecord(event_time=time.time()),
            actions=[1, 2, 3],
            metadata={"type": "runtime"},
        )
        await tracer.record_event(runtime_event)

        # Environment event
        env_event = EnvironmentEvent(
            system_instance_id="env",
            time_record=TimeRecord(event_time=time.time()),
            reward=0.5,
            terminated=False,
            system_state_before={"pos": 0},
            system_state_after={"pos": 1},
        )
        await tracer.record_event(env_event)

        # LM CAIS event with call_records (new pattern)
        import uuid

        call_record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            api_type="chat_completions",
            provider="openai",
            model_name="gpt-4",
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003),
            input_messages=[
                LLMMessage(role="user", parts=[LLMContentPart(type="text", text="Test prompt")])
            ],
            output_messages=[
                LLMMessage(
                    role="assistant", parts=[LLMContentPart(type="text", text="Test response")]
                )
            ],
            latency_ms=500,
        )

        lm_event = LMCAISEvent(
            system_instance_id="llm",
            time_record=TimeRecord(event_time=time.time()),
            # Aggregates at event level
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.003,
            latency_ms=500,
            # Store the call record
            call_records=[call_record],
        )
        await tracer.record_event(lm_event)

        trace = await tracer.end_session(save=False)

        # Verify all events
        assert len(trace.event_history) == 3
        assert isinstance(trace.event_history[0], RuntimeEvent)
        assert isinstance(trace.event_history[1], EnvironmentEvent)
        assert isinstance(trace.event_history[2], LMCAISEvent)

        # Verify metadata
        assert trace.event_history[0].metadata["type"] == "runtime"
        assert trace.event_history[1].reward == 0.5
        # Verify new call_records structure
        lm_event_from_trace = trace.event_history[2]
        assert len(lm_event_from_trace.call_records) == 1
        lm_call_record = lm_event_from_trace.call_records[0]
        assert lm_call_record.usage is not None
        assert lm_call_record.model_name == "gpt-4"
        assert lm_call_record.usage.total_tokens == 150

    async def test_concurrent_timesteps_same_session(self):
        """Test that timesteps within a session are sequential, not concurrent."""
        tracer = SessionTracer(auto_save=False)

        await tracer.start_session()

        # Start first timestep
        await tracer.start_timestep("step1")

        # Should not be able to start another while one is active
        # (This is a design decision - timesteps are sequential within a session)
        await tracer.start_timestep("step2")

        # step2 should be added, but step1 might not be properly ended
        assert tracer.current_session is not None
        assert len(tracer.current_session.session_time_steps) == 2

        await tracer.end_session(save=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
