#!/usr/bin/env python3
"""
Debug render issue after deserialization
========================================
"""

import asyncio
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_render_issue():
    """Test render issue after deserialization."""
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
    
    # Serialize and restore
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    print("After restoration:")
    print(f"  Player exists: {restored_env.engine.env._player is not None}")
    
    # Test render directly
    print("\nTesting render directly:")
    try:
        img = restored_env.engine.env.render()
        print(f"  Render succeeded, shape: {img.shape}")
        print(f"  Player after render: {restored_env.engine.env._player}")
    except Exception as e:
        print(f"  Render failed: {e}")
        print(f"  Player after failed render: {restored_env.engine.env._player}")
    
    # Check what's in _local_view
    print("\n_local_view details:")
    print(f"  Has _local_view: {hasattr(restored_env.engine.env, '_local_view')}")
    if hasattr(restored_env.engine.env, '_local_view'):
        print(f"  _local_view type: {type(restored_env.engine.env._local_view)}")
    
    # Check world state
    print("\nWorld state:")
    print(f"  Has _world: {hasattr(restored_env.engine.env, '_world')}")
    if hasattr(restored_env.engine.env, '_world'):
        print(f"  World has _local_view: {hasattr(restored_env.engine.env._world, '_local_view')}")
        
    # Try to manually fix
    print("\nTrying manual fix:")
    if not hasattr(restored_env.engine.env, '_local_view'):
        # Import necessary modules
        import crafter
        from crafter.engine import LocalView
        restored_env.engine.env._local_view = LocalView(
            restored_env.engine.env._world,
            restored_env.engine.env._view
        )
        print("  Added _local_view")
        
        # Try render again
        try:
            img = restored_env.engine.env.render()
            print(f"  Render now works! Shape: {img.shape}")
        except Exception as e:
            print(f"  Still fails: {e}")


if __name__ == "__main__":
    asyncio.run(test_render_issue())