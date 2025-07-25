#!/usr/bin/env python3
"""
Debug exactly where player reference is lost
============================================
"""

import asyncio
from uuid import uuid4
import gc

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def debug_player_loss():
    """Find exactly where player reference is lost."""
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
    
    # Create and initialize environment
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take some actions
    for action in [1, 2, 5, 3]:
        await env.step(EnvToolCall(tool="interact", args={"action": action}))
    
    print("BEFORE SERIALIZATION:")
    print(f"  Player id: {id(env.engine.env._player)}")
    print(f"  Player: {env.engine.env._player}")
    print(f"  Player pos: {env.engine.env._player.pos}")
    
    # Serialize
    snapshot = await env._serialize_engine()
    
    # Check snapshot contents
    print("\nSNAPSHOT CONTENTS:")
    print(f"  env_raw_state keys: {list(snapshot.env_raw_state.keys())}")
    print(f"  player_idx in snapshot: {snapshot.env_raw_state.get('player_idx')}")
    
    # Deserialize
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    print("\nAFTER DESERIALIZATION:")
    print(f"  Player id: {id(restored_env.engine.env._player)}")
    print(f"  Player: {restored_env.engine.env._player}")
    print(f"  Player pos: {restored_env.engine.env._player.pos if restored_env.engine.env._player else 'None'}")
    
    # Check if player is in objects list
    player_in_objects = False
    for i, obj in enumerate(restored_env.engine.env._world._objects):
        if obj is restored_env.engine.env._player:
            player_in_objects = True
            print(f"  Player found in objects at index {i}")
            break
    if not player_in_objects:
        print("  Player NOT in objects list!")
    
    # Now let's trace through a step
    print("\nTRACING THROUGH STEP:")
    
    # Check before _step_engine
    print("1. Before _step_engine:")
    print(f"   Player: {restored_env.engine.env._player}")
    
    # Check at start of _step_engine (manually)
    print("\n2. Inside _step_engine (manual check):")
    print(f"   self.env._player: {restored_env.engine.env._player}")
    
    # Try render directly
    print("\n3. Testing render directly:")
    try:
        img = restored_env.engine.env.render()
        print(f"   Render succeeded")
        print(f"   Player after render: {restored_env.engine.env._player}")
    except Exception as e:
        print(f"   Render failed: {e}")
        print(f"   Player after failed render: {restored_env.engine.env._player}")
    
    # Now try actual step
    print("\n4. Trying actual step:")
    try:
        result = await restored_env.step(EnvToolCall(tool="interact", args={"action": 5}))
        print(f"   Step succeeded, reward: {result.get('reward_last_step')}")
    except Exception as e:
        print(f"   Step failed: {e}")
    
    # Check player again
    print(f"\n5. Player after step attempt: {restored_env.engine.env._player}")
    
    # Check if garbage collection is involved
    print("\n6. Checking garbage collection:")
    gc.collect()
    print(f"   Player after gc: {restored_env.engine.env._player}")


if __name__ == "__main__":
    asyncio.run(debug_player_loss())