#!/usr/bin/env python3
"""
Test step by step what happens
===============================
"""

import asyncio
from uuid import uuid4
import sys

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


class DebugCrafterEngine:
    """Wrapper to debug the engine."""
    
    def __init__(self, engine):
        self.engine = engine
        
    async def step(self, action):
        """Step with detailed debugging."""
        print(f"\n=== DEBUG STEP {action} ===")
        
        # Check initial state
        print(f"1. Initial player: {self.engine.env._player}")
        
        # Call original step
        try:
            # Manually do what _step_engine does
            if self.engine.env._player is None:
                print("2. Player is None at start of step!")
                # Try to find player
                for i, obj in enumerate(self.engine.env._world._objects):
                    if obj and hasattr(obj, '__class__') and obj.__class__.__name__ == 'Player':
                        print(f"   Found player at index {i}")
                        self.engine.env._player = obj
                        break
            else:
                print("2. Player exists at start of step")
            
            # Try to render
            print("3. Attempting render...")
            try:
                img = self.engine.env.render()
                print("   Render succeeded")
            except Exception as e:
                print(f"   Render failed: {e}")
            
            print(f"4. Player after render: {self.engine.env._player}")
            
            # Do the actual env step
            print("5. Calling env.step...")
            obs, reward, done, info = self.engine.env.step(action)
            print(f"   Step returned: reward={reward}, done={done}")
            
            print(f"6. Player after env.step: {self.engine.env._player}")
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"ERROR in step: {e}")
            import traceback
            traceback.print_exc()
            raise


async def test_step_by_step():
    """Test what happens step by step."""
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
    
    # Create env1 and take actions
    print("Creating env1 and taking actions...")
    env1 = CrafterClassicEnvironment(task)
    await env1.initialize()
    
    for action in [1, 2, 5, 3]:
        await env1.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Serialize
    print("\nSerializing...")
    snapshot = await env1._serialize_engine()
    
    # Deserialize
    print("Deserializing...")
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Wrap engine for debugging
    debug_engine = DebugCrafterEngine(env2.engine)
    
    # Take a step with debugging
    print("\nTaking step with debug wrapper...")
    obs, reward, done, info = await debug_engine.step(5)
    
    print(f"\nFinal state:")
    print(f"  Player: {env2.engine.env._player}")
    print(f"  Reward: {reward}")


if __name__ == "__main__":
    asyncio.run(test_step_by_step())