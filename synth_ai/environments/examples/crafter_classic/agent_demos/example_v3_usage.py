#!/usr/bin/env python3
"""
Simple example demonstrating how to use tracing_v3 with Crafter.
This shows the basic pattern for converting v2 code to v3.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Import v3 tracing components
# Import Crafter hooks for v3
from synth_ai.environments.examples.crafter_classic.trace_hooks_v3 import CRAFTER_HOOKS

# Import LM 
from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v3.abstractions import (
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    TimeRecord,
)
from synth_ai.tracing_v3.decorators import set_session_id, set_turn_number
from synth_ai.tracing_v3.session_tracer import SessionTracer
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager


async def simple_crafter_session():
    """Run a simple Crafter session with v3 tracing."""
    
    # 1. Create session tracer with hooks
    tracer = SessionTracer(
        hooks=CRAFTER_HOOKS,
        db_url="sqlite+libsql://http://127.0.0.1:8080",  # Turso URL
        auto_save=True
    )
    
    # 2. Start a session
    session_id = await tracer.start_session(
        metadata={
            "experiment": "v3_example",
            "model": "gpt-4o-mini",
            "difficulty": "easy"
        }
    )
    print(f"Started session: {session_id}")
    
    # 3. Simulate a few game turns
    for turn in range(5):
        # Start timestep
        await tracer.start_timestep(f"turn_{turn}", turn_number=turn)
        
        # Record observation message
        observation = {
            "inventory": {"wood": turn, "stone": 0},
            "nearby": ["tree", "stone"],
            "status": {"health": 9, "food": 8 - turn}
        }
        
        await tracer.record_message(
            content=json.dumps(observation),
            message_type="observation",
            metadata={"source": "environment"}
        )
        
        # Record LM event (simulated)
        lm_event = LMCAISEvent(
            system_instance_id="crafter_agent",
            time_record=TimeRecord(
                event_time=time.time(),
                message_time=turn
            ),
            model_name="gpt-4o-mini",
            provider="openai",
            input_tokens=100 + turn * 10,
            output_tokens=20,
            total_tokens=120 + turn * 10,
            cost_usd=0.001 * (turn + 1),
            latency_ms=100 + turn * 50
        )
        await tracer.record_event(lm_event)
        
        # Record action
        action = "collect_wood" if turn % 2 == 0 else "move_right"
        await tracer.record_message(
            content=action,
            message_type="action",
            metadata={"source": "agent"}
        )
        
        # Record runtime event
        runtime_event = RuntimeEvent(
            system_instance_id="crafter_env",
            time_record=TimeRecord(
                event_time=time.time(),
                message_time=turn
            ),
            actions=[5 if turn % 2 == 0 else 2],  # action IDs
            metadata={
                "action_name": action,
                "valid": True
            }
        )
        await tracer.record_event(runtime_event)
        
        # Record environment event with achievements
        achievements_before = {"collect_wood": turn > 0}
        achievements_after = {"collect_wood": True} if action == "collect_wood" else achievements_before
        
        env_event = EnvironmentEvent(
            system_instance_id="crafter_env",
            time_record=TimeRecord(
                event_time=time.time(),
                message_time=turn
            ),
            reward=1.0 if action == "collect_wood" else 0.0,
            terminated=False,
            system_state_before={
                "public_state": {"achievements_status": achievements_before}
            },
            system_state_after={
                "public_state": {"achievements_status": achievements_after}
            }
        )
        await tracer.record_event(env_event)
        
        # End timestep
        await tracer.end_timestep()
        
        print(f"Completed turn {turn}")
    
    # 4. End session (auto-saves to database)
    trace = await tracer.end_session()
    print(f"Session ended. Total events: {len(trace.event_history)}")
    
    # 5. Query the saved data
    db_manager = AsyncSQLTraceManager("sqlite+libsql://http://127.0.0.1:8080")
    await db_manager.initialize()
    
    # Get session data
    session_data = await db_manager.get_session_trace(session_id)
    if session_data:
        print(f"\nRetrieved session from database:")
        print(f"  Session ID: {session_data['session_id']}")
        print(f"  Timesteps: {session_data['num_timesteps']}")
        print(f"  Events: {session_data['num_events']}")
        print(f"  Messages: {session_data['num_messages']}")
    
    # Query model usage
    model_usage = await db_manager.get_model_usage()
    print(f"\nModel usage statistics:")
    print(model_usage)
    
    await db_manager.close()
    await tracer.close()


async def context_manager_example():
    """Example using context managers for cleaner code."""
    
    tracer = SessionTracer(
        hooks=CRAFTER_HOOKS,
        db_url="sqlite+libsql://http://127.0.0.1:8080"
    )
    
    # Use context managers for automatic cleanup
    async with tracer.session(metadata={"example": "context_manager"}) as session_id:
        print(f"In session: {session_id}")
        
        async with tracer.timestep("step_1", turn_number=0) as step:
            print(f"In timestep: {step.step_id}")
            
            # Record some events
            await tracer.record_message(
                content="Hello from context manager",
                message_type="user"
            )
            
            event = RuntimeEvent(
                system_instance_id="example_system",
                time_record=TimeRecord(event_time=time.time()),
                actions=[1],
                metadata={"example": True}
            )
            await tracer.record_event(event)
    
    print("Session automatically ended and saved")
    await tracer.close()


async def main():
    """Run the examples."""
    print("=== V3 Tracing Examples ===\n")
    
    print("1. Running simple Crafter session...")
    await simple_crafter_session()
    
    print("\n2. Running context manager example...")
    await context_manager_example()
    
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    # Make sure sqld is running on port 8080
    print("Note: This example assumes sqld is running on http://127.0.0.1:8080")
    print("Start it with: sqld --http-listen 127.0.0.1:8080\n")
    
    asyncio.run(main())