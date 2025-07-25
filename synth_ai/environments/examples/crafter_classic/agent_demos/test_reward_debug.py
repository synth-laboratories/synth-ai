#!/usr/bin/env python3
"""
Debug reward calculation after restore
======================================
"""

import asyncio
import numpy as np
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_reward_system():
    """Debug reward calculation."""
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test rewards"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=CrafterTaskInstanceMetadata(
            difficulty="easy",
            seed=600,
            num_trees_radius=5,
            num_cows_radius=2,
            num_hostiles_radius=0
        ),
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    env1 = CrafterClassicEnvironment(task)
    await env1.initialize()
    
    # Take a few steps
    for i in range(3):
        await env1.step(EnvToolCall(tool="interact", args={"action": i}))
    
    print("Before save:")
    print(f"  _total_reward: {env1.engine._total_reward}")
    print(f"  Has reward_stack: {hasattr(env1.engine, 'reward_stack')}")
    print(f"  Has _reward_stack: {hasattr(env1.engine, '_reward_stack')}")
    print(f"  Has achievements_unlocked: {hasattr(env1.engine, 'achievements_unlocked')}")
    print(f"  Achievements unlocked: {getattr(env1.engine, 'achievements_unlocked', 'N/A')}")
    
    # Save
    snapshot = await env1._serialize_engine()
    
    # Restore
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    print("\nAfter restore:")
    print(f"  _total_reward: {env2.engine._total_reward}")
    print(f"  Has reward_stack: {hasattr(env2.engine, 'reward_stack')}")
    print(f"  Has _reward_stack: {hasattr(env2.engine, '_reward_stack')}")
    print(f"  Has achievements_unlocked: {hasattr(env2.engine, 'achievements_unlocked')}")
    print(f"  Achievements unlocked: {getattr(env2.engine, 'achievements_unlocked', 'N/A')}")
    
    # Test step with debugging
    print("\nTaking a step...")
    
    # Check step count
    print(f"  Current step: {env2.engine.env._step}")
    print(f"  Max steps: {env2.engine.env._length}")
    
    # Take step
    action = 5
    print(f"  Action: {action}")
    
    # Manually step to see what happens
    try:
        obs, reward, done, info = env2.engine.env.step(action)
        print(f"  Raw step result: reward={reward}, done={done}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Now try through engine
        result = await env2.step(EnvToolCall(tool="interact", args={"action": action}))
        print(f"  Engine step result: reward={result.get('reward_last_step', 'N/A')}")
        
    except Exception as e:
        print(f"  Step failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_reward_system())