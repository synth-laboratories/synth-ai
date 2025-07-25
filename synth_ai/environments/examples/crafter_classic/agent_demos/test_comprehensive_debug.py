#!/usr/bin/env python3
"""
Debug the comprehensive test failure
===================================
"""

import asyncio
import numpy as np
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_deterministic_gameplay_after_restore():
    """Test that gameplay is completely deterministic after restore."""
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
    initial_actions = [1, 2, 5, 3, 5, 0, 2, 5]
    for action in initial_actions[:4]:
        await env1.step(EnvToolCall(tool="interact", args={"action": action}))
    
    print("Before serialization:")
    print(f"  env._player: {env1.engine.env._player}")
    print(f"  Player pos: {env1.engine.env._player.pos if env1.engine.env._player else 'None'}")
    
    # Save state
    snapshot = await env1._serialize_engine()
    
    print("\nSnapshot info:")
    print(f"  total_reward: {snapshot.total_reward_snapshot}")
    print(f"  player_idx in saved state: {snapshot.env_raw_state.get('player_idx', 'Not found')}")
    
    # Continue in env1
    results1 = []
    for action in initial_actions[4:]:
        obs = await env1.step(EnvToolCall(tool="interact", args={"action": action}))
        pub = env1.engine._get_public_state_from_env()
        results1.append({
            'position': pub.player_position,
            'reward': obs.get('reward_last_step', 0),
        })
    
    print(f"\nEnv1 results: {results1}")
    
    # Create new env and restore
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    print("\nAfter deserialization:")
    print(f"  env._player: {env2.engine.env._player}")
    print(f"  Player pos: {env2.engine.env._player.pos if env2.engine.env._player else 'None'}")
    print(f"  Can render: ", end="")
    try:
        _ = env2.engine.env.render()
        print("Yes")
    except Exception as e:
        print(f"No - {e}")
    
    # Take same actions
    results2 = []
    for i, action in enumerate(initial_actions[4:]):
        print(f"\nAbout to take action {action} (step {i})...")
        print(f"  Player before step: {env2.engine.env._player}")
        
        obs = await env2.step(EnvToolCall(tool="interact", args={"action": action}))
        
        print(f"  Player after step: {env2.engine.env._player}")
        print(f"  Reward: {obs.get('reward_last_step', 0)}")
        
        pub = env2.engine._get_public_state_from_env()
        results2.append({
            'position': pub.player_position,
            'reward': obs.get('reward_last_step', 0),
        })
        
        if obs.get('reward_last_step', 0) == -1.0:
            print(f"  Error info: {pub.error_info}")
            break
    
    print(f"\nEnv2 results: {results2}")


if __name__ == "__main__":
    asyncio.run(test_deterministic_gameplay_after_restore())