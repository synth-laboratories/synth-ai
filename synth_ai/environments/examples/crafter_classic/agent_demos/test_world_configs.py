#!/usr/bin/env python3
"""
Test different world configurations
===================================
"""

import asyncio
import json
from uuid import uuid4
from pathlib import Path

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


async def test_world_config(config_name: str, seed: int = 42):
    """Test a specific world configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {config_name.upper()} world configuration")
    print(f"{'='*60}")
    
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions=f"Test {config_name} world"),
        intent=Intent(rubric={"goal": "Explore"}, gold_trajectories=None, gold_state_diff={}),
        metadata=CrafterTaskInstanceMetadata(
            difficulty=config_name,
            seed=seed,
            num_trees_radius=0,  # These are now controlled by world_config
            num_cows_radius=0,
            num_hostiles_radius=0,
            world_config=config_name
        ),
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Count entities near spawn
    world = env.engine.env._world
    player_pos = env.engine.env._player.pos
    
    counts = {
        'trees': 0,
        'cows': 0,
        'zombies': 0,
        'skeletons': 0,
        'coal': 0,
        'iron': 0,
        'diamonds': 0
    }
    
    # Count objects in 20x20 area around spawn
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            x, y = player_pos[0] + dx, player_pos[1] + dy
            if 0 <= x < world.area[0] and 0 <= y < world.area[1]:
                # Check material
                mat_id = world._mat_map[x, y]
                if mat_id == world._mat_ids['coal']:
                    counts['coal'] += 1
                elif mat_id == world._mat_ids['iron']:
                    counts['iron'] += 1
                elif mat_id == world._mat_ids['diamond']:
                    counts['diamonds'] += 1
                
                # Check objects
                for obj in world._objects:
                    if obj and hasattr(obj, 'pos') and all(obj.pos == [x, y]):
                        if obj.__class__.__name__ == 'Tree':
                            counts['trees'] += 1
                        elif obj.__class__.__name__ == 'Cow':
                            counts['cows'] += 1
                        elif obj.__class__.__name__ == 'Zombie':
                            counts['zombies'] += 1
                        elif obj.__class__.__name__ == 'Skeleton':
                            counts['skeletons'] += 1
    
    print(f"\nResources near spawn (20x20 area):")
    print(f"  Trees: {counts['trees']}")
    print(f"  Coal: {counts['coal']}")
    print(f"  Iron: {counts['iron']}")
    print(f"  Diamonds: {counts['diamonds']}")
    
    print(f"\nEntities near spawn:")
    print(f"  Cows: {counts['cows']}")
    print(f"  Zombies: {counts['zombies']}")
    print(f"  Skeletons: {counts['skeletons']}")
    
    # Take some steps to see dynamic spawning
    print(f"\nTaking 50 steps to observe dynamic spawning...")
    for _ in range(50):
        await env.step(EnvToolCall(tool="interact", args={"action": 0}))  # noop
    
    # Count again
    new_counts = {'cows': 0, 'zombies': 0, 'skeletons': 0}
    for obj in world._objects:
        if obj:
            if obj.__class__.__name__ == 'Cow':
                new_counts['cows'] += 1
            elif obj.__class__.__name__ == 'Zombie':
                new_counts['zombies'] += 1
            elif obj.__class__.__name__ == 'Skeleton':
                new_counts['skeletons'] += 1
    
    print(f"\nTotal entities after 50 steps:")
    print(f"  Cows: {counts['cows']} → {new_counts['cows']} ({new_counts['cows'] - counts['cows']:+d})")
    print(f"  Zombies: {counts['zombies']} → {new_counts['zombies']} ({new_counts['zombies'] - counts['zombies']:+d})")
    print(f"  Skeletons: {counts['skeletons']} → {new_counts['skeletons']} ({new_counts['skeletons'] - counts['skeletons']:+d})")


async def test_custom_config():
    """Test a custom JSON configuration."""
    print(f"\n{'='*60}")
    print(f"Testing CUSTOM JSON world configuration")
    print(f"{'='*60}")
    
    # Create a custom config
    custom_config = {
        "name": "Ultra Resources",
        "description": "Tons of resources, no enemies",
        "materials": {
            "coal": {"threshold": -0.5, "probability": 0.5},
            "iron": {"threshold": -0.2, "probability": 0.6},
            "diamond": {"threshold": 0.0, "probability": 0.1},
            "tree": {"threshold": -0.5, "probability": 0.6},
            "lava": {"threshold": 0.8, "probability": 1.0}
        },
        "initial_spawns": {
            "cow": {"min_distance": 1, "probability": 0.1},
            "zombie": {"min_distance": 100, "probability": 0},
            "skeleton": {"probability": 0}
        },
        "dynamic_spawns": {
            "zombie": {"count_range": [0, 0], "min_distance": 100, "spawn_prob": 0, "despawn_prob": 1},
            "skeleton": {"count_range": [0, 0], "min_distance": 100, "spawn_prob": 0, "despawn_prob": 1},
            "cow": {"count_range": [0, 10], "min_distance": 1, "spawn_prob": 0.1, "despawn_prob": 0}
        }
    }
    
    # Save config
    config_path = Path("ultra_resources_config.json")
    with open(config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test custom world"),
        intent=Intent(rubric={"goal": "Explore"}, gold_trajectories=None, gold_state_diff={}),
        metadata=CrafterTaskInstanceMetadata(
            difficulty="custom",
            seed=42,
            num_trees_radius=0,
            num_cows_radius=0,
            num_hostiles_radius=0,
            world_config_path=str(config_path)
        ),
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Count resources
    world = env.engine.env._world
    total_resources = {'coal': 0, 'iron': 0, 'diamond': 0, 'trees': 0}
    
    # Count in entire world
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            mat_id = world._mat_map[x, y]
            if mat_id == world._mat_ids['coal']:
                total_resources['coal'] += 1
            elif mat_id == world._mat_ids['iron']:
                total_resources['iron'] += 1
            elif mat_id == world._mat_ids['diamond']:
                total_resources['diamond'] += 1
    
    for obj in world._objects:
        if obj and obj.__class__.__name__ == 'Tree':
            total_resources['trees'] += 1
    
    print(f"\nTotal resources in world (64x64):")
    print(f"  Coal blocks: {total_resources['coal']}")
    print(f"  Iron blocks: {total_resources['iron']}")
    print(f"  Diamond blocks: {total_resources['diamond']}")
    print(f"  Trees: {total_resources['trees']}")
    
    # Clean up
    config_path.unlink()


async def main():
    """Test all world configurations."""
    print("Testing Crafter World Configurations")
    print("=" * 60)
    
    # Test predefined configs
    for config in ["easy", "normal", "hard", "peaceful"]:
        await test_world_config(config)
    
    # Test custom config
    await test_custom_config()
    
    print(f"\n{'='*60}")
    print("All tests complete!")
    print("\nSummary:")
    print("- EASY: More resources, fewer enemies, safer spawning")
    print("- NORMAL: Standard Crafter experience")
    print("- HARD: Scarce resources, many enemies, dangerous")
    print("- PEACEFUL: No enemies, abundant resources")
    print("- CUSTOM: Fully configurable via JSON")


if __name__ == "__main__":
    asyncio.run(main())