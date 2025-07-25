#!/usr/bin/env python3
"""
Minimal test for determinism issue
==================================
"""

import asyncio
import numpy as np
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_minimal():
    """Minimal test to isolate the issue."""
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test"),
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
    
    # Env 1: Take actions directly
    env1 = CrafterClassicEnvironment(task)
    await env1.initialize()
    
    # Actions
    actions = [1, 2, 5, 3, 5, 0, 2, 5]
    
    # Take all actions in env1
    results1 = []
    for i, action in enumerate(actions):
        obs = await env1.step(EnvToolCall(tool="interact", args={"action": action}))
        results1.append(obs.get('reward_last_step', 0))
        print(f"Env1 step {i}: action={action}, reward={obs.get('reward_last_step', 0):.3f}")
    
    print(f"\nEnv1 total rewards: {results1}")
    
    # Env 2: Take first 4 actions, save, restore, continue
    env2 = CrafterClassicEnvironment(task)
    await env2.initialize()
    
    # Take first 4 actions
    for i in range(4):
        await env2.step(EnvToolCall(tool="interact", args={"action": actions[i]}))
    
    # Save and restore
    snapshot = await env2._serialize_engine()
    env3 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Continue with remaining actions
    results2 = []
    for i in range(4, 8):
        obs = await env3.step(EnvToolCall(tool="interact", args={"action": actions[i]}))
        results2.append(obs.get('reward_last_step', 0))
        print(f"Env3 step {i-4}: action={actions[i]}, reward={obs.get('reward_last_step', 0):.3f}")
    
    print(f"\nEnv3 rewards for steps 4-7: {results2}")
    print(f"Expected rewards: {results1[4:8]}")
    print(f"Match: {results2 == results1[4:8]}")


if __name__ == "__main__":
    asyncio.run(test_minimal())