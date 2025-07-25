#!/usr/bin/env python3
"""
Debug deterministic gameplay with better error handling
======================================================
"""

import asyncio
import traceback
import numpy as np
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_with_error_handling():
    """Test with detailed error tracking."""
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test determinism"),
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
    
    # Take initial actions
    initial_actions = [1, 2, 5, 3]
    for action in initial_actions:
        await env1.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Check engine state before save
    print("Before save:")
    print(f"  env._player: {env1.engine.env._player}")
    print(f"  player in objects: {env1.engine.env._player in env1.engine.env._world._objects}")
    print(f"  total objects: {len([o for o in env1.engine.env._world._objects if o is not None])}")
    
    # Save state
    snapshot = await env1._serialize_engine()
    
    # Restore
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Check engine state after restore
    print("\nAfter restore:")
    print(f"  env._player: {env2.engine.env._player}")
    print(f"  player in objects: {env2.engine.env._player in env2.engine.env._world._objects}")
    print(f"  total objects: {len([o for o in env2.engine.env._world._objects if o is not None])}")
    
    # Try a step with detailed error tracking
    try:
        print("\nAttempting step...")
        # Check internal state
        print(f"  env.obs exists: {hasattr(env2.engine, 'obs')}")
        print(f"  env.done exists: {hasattr(env2.engine, 'done')}")
        print(f"  env.info exists: {hasattr(env2.engine, 'info')}")
        
        # Manually initialize if needed
        if not hasattr(env2.engine, 'obs'):
            print("  Initializing missing attributes...")
            env2.engine.obs = env2.engine.env.render()
            env2.engine.done = False
            env2.engine.info = {}
            env2.engine.last_reward = 0.0
            
        obs = await env2.step(EnvToolCall(tool="interact", args={"action": 5}))
        print(f"Step succeeded! Reward: {obs.get('reward_last_step', 0)}")
        
    except Exception as e:
        print(f"Step failed with error: {e}")
        traceback.print_exc()
        
        # Check public state for error info
        try:
            pub = env2.engine._get_public_state_from_env()
            if pub.error_info:
                print(f"Public state error: {pub.error_info}")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_with_error_handling())