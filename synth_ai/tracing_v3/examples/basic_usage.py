"""Basic usage example for tracing v3."""

import asyncio
import time
from typing import Any

from .. import SessionTracer
from ..abstractions import EnvironmentEvent, LMCAISEvent, RuntimeEvent, TimeRecord
from ..turso.daemon import SqldDaemon


async def simulate_llm_call(model: str, prompt: str) -> dict[str, Any]:
    """Simulate an LLM API call."""
    await asyncio.sleep(0.1)  # Simulate network latency

    # Simulate response
    tokens = len(prompt.split()) * 3
    return {
        "model": model,
        "response": f"Response to: {prompt[:50]}...",
        "usage": {
            "prompt_tokens": len(prompt.split()) * 2,
            "completion_tokens": tokens,
            "total_tokens": len(prompt.split()) * 2 + tokens,
        },
    }


async def main():
    """Demonstrate basic tracing v3 usage."""
    print("Starting tracing v3 example...")

    # Option 1: Start sqld daemon programmatically
    with SqldDaemon():
        print("✓ Started sqld daemon")

        # Wait for daemon to be ready
        await asyncio.sleep(1)

        # Create tracer
        tracer = SessionTracer()
        await tracer.initialize()
        print("✓ Initialized tracer")

        # Example 1: Basic session with events
        print("\n--- Example 1: Basic Session ---")
        async with tracer.session(metadata={"example": "basic"}) as session_id:
            print(f"Started session: {session_id}")

            # Timestep 1: LLM interaction
            async with tracer.timestep("llm_step", turn_number=1):
                # Simulate LLM call
                result = await simulate_llm_call("gpt-4", "What is the capital of France?")

                # Record LLM event
                event = LMCAISEvent(
                    system_instance_id="llm_system",
                    time_record=TimeRecord(event_time=time.time()),
                    model_name=result["model"],
                    input_tokens=result["usage"]["prompt_tokens"],
                    output_tokens=result["usage"]["completion_tokens"],
                    total_tokens=result["usage"]["total_tokens"],
                    cost_usd=0.003,  # $0.003
                    latency_ms=100,
                    metadata={"prompt": "What is the capital of France?"},
                )
                await tracer.record_event(event)

                # Record messages
                await tracer.record_message(
                    content="What is the capital of France?", message_type="user"
                )
                await tracer.record_message(content=result["response"], message_type="assistant")
                print("✓ Recorded LLM interaction")

            # Timestep 2: Environment interaction
            async with tracer.timestep("env_step", turn_number=2):
                # Record environment event
                env_event = EnvironmentEvent(
                    system_instance_id="environment",
                    time_record=TimeRecord(event_time=time.time()),
                    reward=0.8,
                    terminated=False,
                    system_state_before={"position": [0, 0]},
                    system_state_after={"position": [1, 0]},
                )
                await tracer.record_event(env_event)
                print("✓ Recorded environment event")

            # Timestep 3: Runtime action
            async with tracer.timestep("runtime_step", turn_number=3):
                # Record runtime event
                runtime_event = RuntimeEvent(
                    system_instance_id="agent",
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[1, 0, 0, 1],  # Example action vector
                    metadata={"action_type": "move_right"},
                )
                await tracer.record_event(runtime_event)
                print("✓ Recorded runtime event")

        print(f"✓ Session {session_id} saved\n")

        # Example 2: Concurrent sessions
        print("--- Example 2: Concurrent Sessions ---")

        async def run_concurrent_session(session_num: int):
            """Run a session concurrently."""
            async with tracer.session(
                metadata={"example": "concurrent", "session_num": session_num}
            ) as sid:
                for i in range(3):
                    async with tracer.timestep(f"step_{i}", turn_number=i):
                        # Simulate some work
                        await asyncio.sleep(0.05)

                        # Record event
                        event = RuntimeEvent(
                            system_instance_id=f"worker_{session_num}",
                            time_record=TimeRecord(event_time=time.time()),
                            actions=[i],
                            metadata={"iteration": i},
                        )
                        await tracer.record_event(event)

                return sid

        # Run 5 concurrent sessions
        tasks = [run_concurrent_session(i) for i in range(5)]
        session_ids = await asyncio.gather(*tasks)
        print(f"✓ Completed {len(session_ids)} concurrent sessions")

        # Example 3: Query stored data
        print("\n--- Example 3: Querying Data ---")

        # Get model usage statistics
        if tracer.db is None:
            raise RuntimeError("Tracer database backend is not initialized")

        model_usage = await tracer.db.get_model_usage()
        print("\nModel Usage:")
        print(model_usage)

        # Query recent sessions
        recent_sessions = await tracer.get_session_history(limit=5)
        print(f"\nRecent Sessions: {len(recent_sessions)} found")
        for session in recent_sessions:
            print(
                f"  - {session['session_id']}: "
                f"{session['num_events']} events, "
                f"{session['num_messages']} messages"
            )

        # Get specific session details
        if recent_sessions:
            session_detail = await tracer.db.get_session_trace(recent_sessions[0]["session_id"])
            if session_detail:
                print(f"\nSession Detail for {session_detail['session_id']}:")
                print(f"  Created: {session_detail['created_at']}")
                print(f"  Timesteps: {len(session_detail['timesteps'])}")

        # Example 4: Using hooks
        print("\n--- Example 4: Hooks ---")

        # Add a custom hook
        call_count = {"count": 0}

        async def count_events(event, **kwargs):
            call_count["count"] += 1
            print(f"  Hook: Event #{call_count['count']} recorded")

        tracer.hooks.register("event_recorded", count_events, name="event_counter")

        async with (
            tracer.session(metadata={"example": "hooks"}) as session_id,
            tracer.timestep("hook_test"),
        ):
            for i in range(3):
                event = RuntimeEvent(
                    system_instance_id="hook_test",
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[i],
                )
                await tracer.record_event(event)

        print(f"✓ Hook called {call_count['count']} times")

        # Cleanup
        await tracer.close()
        print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
