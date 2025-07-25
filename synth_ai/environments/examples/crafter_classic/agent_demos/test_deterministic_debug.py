#!/usr/bin/env python3
"""
Debug deterministic gameplay after restore
==========================================
"""

import asyncio
import numpy as np
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_deterministic_gameplay_debug():
    """Debug why gameplay isn't deterministic after restore."""
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
    print(f"Taking first 4 actions: {initial_actions[:4]}")
    
    for i, action in enumerate(initial_actions[:4]):
        obs = await env1.step(EnvToolCall(tool="interact", args={"action": action}))
        print(f"Step {i}: action={action}, reward={obs.get('reward_last_step', 0):.3f}")
    
    # Save state
    print("\n=== Saving state ===")
    snapshot = await env1._serialize_engine()
    
    # Continue in env1
    print("\n=== Continuing in env1 ===")
    results1 = []
    for i, action in enumerate(initial_actions[4:]):
        obs = await env1.step(EnvToolCall(tool="interact", args={"action": action}))
        pub = env1.engine._get_public_state_from_env()
        reward = obs.get('reward_last_step', 0)
        print(f"Env1 Step {i}: action={action}, reward={reward:.3f}, pos={pub.player_position}")
        results1.append({
            'action': action,
            'position': pub.player_position,
            'inventory': dict(pub.inventory),
            'reward': reward,
            'total_reward': env1.engine._total_reward,
        })
    
    # Create new env and restore
    print("\n=== Restoring to env2 ===")
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Check state immediately after restore
    pub_restored = env2.engine._get_public_state_from_env()
    print(f"Restored position: {pub_restored.player_position}")
    print(f"Restored total reward: {env2.engine._total_reward}")
    
    # Take same actions
    print("\n=== Taking same actions in env2 ===")
    results2 = []
    for i, action in enumerate(initial_actions[4:]):
        obs = await env2.step(EnvToolCall(tool="interact", args={"action": action}))
        pub = env2.engine._get_public_state_from_env()
        reward = obs.get('reward_last_step', 0)
        print(f"Env2 Step {i}: action={action}, reward={reward:.3f}, pos={pub.player_position}")
        results2.append({
            'action': action,
            'position': pub.player_position,
            'inventory': dict(pub.inventory),
            'reward': reward,
            'total_reward': env2.engine._total_reward,
        })
    
    # Compare results
    print("\n=== Comparing results ===")
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        print(f"\nStep {i}:")
        print(f"  Action: {r1['action']} (same: {r1['action'] == r2['action']})")
        print(f"  Position: {r1['position']} vs {r2['position']} (same: {r1['position'] == r2['position']})")
        print(f"  Reward: {r1['reward']:.3f} vs {r2['reward']:.3f} (same: {abs(r1['reward'] - r2['reward']) < 0.001})")
        print(f"  Total reward: {r1['total_reward']:.3f} vs {r2['total_reward']:.3f}")
        
        # Check inventory differences
        inv_diff = False
        for item in set(r1['inventory'].keys()) | set(r2['inventory'].keys()):
            v1 = r1['inventory'].get(item, 0)
            v2 = r2['inventory'].get(item, 0)
            if v1 != v2:
                print(f"  Inventory {item}: {v1} vs {v2}")
                inv_diff = True
        if not inv_diff:
            print("  Inventory: same")


if __name__ == "__main__":
    asyncio.run(test_deterministic_gameplay_debug())