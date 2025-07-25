#!/usr/bin/env python3
"""
Debug what happens during env.step
==================================
"""

import asyncio
import numpy as np
from uuid import uuid4
import pickle
import gzip

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_step_behavior():
    """Test what happens during step after restore."""
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test step"),
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
    
    # Take a few actions
    for action in [1, 2, 5, 3]:
        await env1.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Save state using raw save/load
    print("Saving state...")
    state_dict = env1.engine.env.save()
    
    # Create fresh env
    env2 = CrafterClassicEnvironment(task)
    await env2.initialize()
    
    # Load state
    print("Loading state...")
    env2.engine.env.load(state_dict)
    
    print(f"\nAfter load:")
    print(f"  Player: {env2.engine.env._player}")
    print(f"  Player in objects: {env2.engine.env._player in env2.engine.env._world._objects}")
    
    # Test raw step
    print("\nTesting raw env.step()...")
    print(f"  Before step - player: {env2.engine.env._player}")
    
    try:
        obs, reward, done, info = env2.engine.env.step(5)
        print(f"  After step - player: {env2.engine.env._player}")
        print(f"  Step result: reward={reward}, done={done}")
        
        # Check if player moved
        if env2.engine.env._player:
            print(f"  Player pos: {env2.engine.env._player.pos}")
    except Exception as e:
        print(f"  Step failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test through CrafterEngine
    print("\n\nTesting through new CrafterEngine...")
    env3 = CrafterClassicEnvironment(task)
    await env3.initialize()
    env3.engine.env.load(state_dict)
    
    # Initialize engine state attributes
    env3.engine.obs = env3.engine.env.render()
    env3.engine.done = False
    env3.engine.info = {}
    env3.engine.last_reward = 0.0
    env3.engine.achievements_unlocked = set()
    env3.engine._total_reward = 0.0
    
    print(f"\nBefore engine step:")
    print(f"  Player: {env3.engine.env._player}")
    
    try:
        result = await env3.step(EnvToolCall(tool="interact", args={"action": 5}))
        print(f"\nAfter engine step:")
        print(f"  Player: {env3.engine.env._player}")
        print(f"  Reward: {result.get('reward_last_step', 'N/A')}")
    except Exception as e:
        print(f"  Engine step failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_step_behavior())