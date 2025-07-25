#!/usr/bin/env python3
"""
Simple test of world configurations
===================================
"""

import asyncio
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_config(config_name: str):
    """Test a specific configuration."""
    print(f"\nTesting {config_name} configuration...")
    
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions=f"Test {config_name}"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=CrafterTaskInstanceMetadata(
            difficulty=config_name,
            seed=42,
            num_trees_radius=0,
            num_cows_radius=0,
            num_hostiles_radius=0,
            world_config=config_name
        ),
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    try:
        env = CrafterClassicEnvironment(task)
        await env.initialize()
        
        # Get world info
        world = env.engine.env._world
        player_pos = env.engine.env._player.pos
        
        print(f"  ✓ Environment created successfully")
        print(f"  Player position: {player_pos}")
        print(f"  World config used: {env.engine.env._world_config_name}")
        
        # Count nearby entities
        zombies = 0
        cows = 0
        for obj in world._objects:
            if obj and obj.__class__.__name__ == 'Zombie':
                zombies += 1
            elif obj and obj.__class__.__name__ == 'Cow':
                cows += 1
        
        print(f"  Initial zombies: {zombies}")
        print(f"  Initial cows: {cows}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("Testing World Configurations")
    print("=" * 40)
    
    for config in ["easy", "normal", "hard", "peaceful"]:
        await test_config(config)
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())