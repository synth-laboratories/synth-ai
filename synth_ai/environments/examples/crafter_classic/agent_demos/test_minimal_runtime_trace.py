#!/usr/bin/env python3
"""
Minimal test to verify runtime event tracing works correctly
"""

import asyncio
import json
from pathlib import Path
import uuid

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.tracing_v2.session_tracer import SessionTracer


async def main():
    print("🧪 Minimal Runtime Tracing Test")
    print("=" * 40)
    
    # Create tracer
    traces_dir = Path(__file__).parent / "minimal_traces"
    traces_dir.mkdir(exist_ok=True)
    tracer = SessionTracer(str(traces_dir))
    
    # Start session
    session_id = f"minimal_test_{uuid.uuid4().hex[:8]}"
    tracer.start_session(session_id)
    tracer.add_session_metadata("test_type", "minimal_runtime_trace")
    
    # Create task instance
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=42,
        num_trees_radius=5,
        num_cows_radius=2,
        num_hostiles_radius=0,
    )
    
    task_instance = CrafterTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Test runtime tracing"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )
    
    # Create environment with tracer
    print("✅ Creating environment with tracer...")
    env = CrafterClassicEnvironment(task_instance, session_tracer=tracer)
    
    # Initialize (with timestep)
    tracer.start_timestep("init")
    obs = await env.initialize()
    print(f"✅ Initialized at position: {obs.get('player_position')}")
    
    # Do one action
    tracer.start_timestep("move_right")
    tool_call = EnvToolCall(tool="interact", args={"action": 2})  # move_right
    
    # This will trigger all three runtime events:
    # 1. Tool call validation
    # 2. Tool execution
    # 3. Observation generation
    obs = await env.step(tool_call)
    print(f"✅ Moved to position: {obs.get('player_position')}")
    
    # Save trace
    tracer.end_session(save=True)
    print(f"✅ Trace saved to: {traces_dir / f'{session_id}.json'}")
    
    # Load and check
    with open(traces_dir / f"{session_id}.json", 'r') as f:
        trace_data = json.load(f)
    
    runtime_events = [e for e in trace_data.get('event_history', []) if e.get('event_type') == 'RuntimeEvent']
    print(f"\n📊 Captured {len(runtime_events)} runtime events:")
    
    for event in runtime_events:
        system_id = event.get('system_instance_id', 'unknown')
        metadata = event.get('metadata', {})
        step_type = (metadata.get('validation_step') or 
                    metadata.get('execution_step') or 
                    metadata.get('observation_step', 'unknown'))
        print(f"  - {system_id}: {step_type}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())