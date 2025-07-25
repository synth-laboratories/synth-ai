#!/usr/bin/env python3
"""
Debug player state during serialization
======================================
"""

import asyncio
from uuid import uuid4
import crafter

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_player_serialization():
    """Test player serialization in detail."""
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
    
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take a few actions
    for action in [1, 2, 5, 3]:
        await env.step(EnvToolCall(tool="interact", args={"action": action}))
    
    print("Before save:")
    print(f"  env._player: {env.engine.env._player}")
    print(f"  Player health: {env.engine.env._player.health}")
    print(f"  Player pos: {env.engine.env._player.pos}")
    print(f"  Player id: {id(env.engine.env._player)}")
    
    # Find player in objects
    player_idx = None
    for i, obj in enumerate(env.engine.env._world._objects):
        if obj is env.engine.env._player:
            player_idx = i
            print(f"  Player found at index: {i}")
            break
    
    # Direct save/load test
    print("\nDirect save/load test:")
    state_dict = env.engine.env.save()
    print(f"  Saved player_idx: {state_dict['player_idx']}")
    
    # Create new env and load
    new_env = crafter.Env(area=(64, 64), length=10000, seed=600)
    new_env.reset()
    new_env.load(state_dict)
    
    print(f"\nAfter load:")
    print(f"  new_env._player: {new_env._player}")
    if new_env._player:
        print(f"  Player health: {new_env._player.health}")
        print(f"  Player pos: {new_env._player.pos}")
        
        # Test render
        try:
            _ = new_env.render()
            print("  Render works: Yes")
        except Exception as e:
            print(f"  Render works: No - {e}")
    
    # Now test through full deserialization
    print("\n\nFull deserialization test:")
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    print(f"  restored env._player: {restored_env.engine.env._player}")
    if restored_env.engine.env._player:
        print(f"  Player health: {restored_env.engine.env._player.health}")
        print(f"  Player pos: {restored_env.engine.env._player.pos}")
    
    # Test a step
    print("\nTesting step after restoration:")
    try:
        result = await restored_env.step(EnvToolCall(tool="interact", args={"action": 5}))
        print(f"  Step reward: {result.get('reward_last_step', 'N/A')}")
        print(f"  Player after step: {restored_env.engine.env._player}")
    except Exception as e:
        print(f"  Step failed: {e}")
        
        # Check what happened to player
        print(f"\n  Debugging player state:")
        print(f"    env._player is None: {restored_env.engine.env._player is None}")
        
        # Look for player in objects
        found_player = False
        for i, obj in enumerate(restored_env.engine.env._world._objects):
            if obj and hasattr(obj, '__class__') and obj.__class__.__name__ == 'Player':
                print(f"    Found Player object at index {i}")
                print(f"      Health: {obj.health}")
                print(f"      Pos: {obj.pos}")
                found_player = True
                break
        
        if not found_player:
            print("    No Player object found in world._objects!")


if __name__ == "__main__":
    asyncio.run(test_player_serialization())