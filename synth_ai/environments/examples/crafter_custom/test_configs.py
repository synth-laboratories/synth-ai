#!/usr/bin/env python3
"""
Test custom Crafter configurations
==================================
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from crafter import Env, WorldGenConfig


def test_configuration(config_name: str):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {config_name.upper()} configuration")
    print(f"{'='*60}")
    
    # Create environment with configuration
    env = Env(seed=42, world_config=config_name)
    
    # Get initial observation
    obs = env.reset()
    print(f"Environment created successfully!")
    print(f"World shape: {obs.shape}")
    print(f"Configuration: {config_name}")
    
    # Count initial entities
    world = env._world
    player = env._player
    
    zombies = 0
    skeletons = 0
    cows = 0
    trees = 0
    
    for obj in world._objects:
        if obj is None:
            continue
        obj_name = obj.__class__.__name__
        if obj_name == 'Zombie':
            zombies += 1
        elif obj_name == 'Skeleton':
            skeletons += 1
        elif obj_name == 'Cow':
            cows += 1
        elif obj_name == 'Tree':
            trees += 1
    
    print(f"\nInitial entity counts:")
    print(f"  Zombies: {zombies}")
    print(f"  Skeletons: {skeletons}")
    print(f"  Cows: {cows}")
    print(f"  Trees: {trees}")
    
    # Count resources near spawn (20x20 area)
    px, py = player.pos
    resources = {'coal': 0, 'iron': 0, 'diamond': 0}
    
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            x, y = px + dx, py + dy
            if 0 <= x < world.area[0] and 0 <= y < world.area[1]:
                mat_id = world._mat_map[x, y]
                mat_name = None
                for name, id_val in world._mat_ids.items():
                    if id_val == mat_id:
                        mat_name = name
                        break
                if mat_name in resources:
                    resources[mat_name] += 1
    
    print(f"\nResources near spawn (20x20 area):")
    for resource, count in resources.items():
        print(f"  {resource}: {count}")
    
    # Take some steps to see dynamic spawning
    print(f"\nSimulating 100 steps...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # Count entities again
    new_zombies = sum(1 for obj in world._objects if obj and obj.__class__.__name__ == 'Zombie')
    new_skeletons = sum(1 for obj in world._objects if obj and obj.__class__.__name__ == 'Skeleton')
    new_cows = sum(1 for obj in world._objects if obj and obj.__class__.__name__ == 'Cow')
    
    print(f"\nEntity counts after 100 steps:")
    print(f"  Zombies: {zombies} → {new_zombies} ({new_zombies - zombies:+d})")
    print(f"  Skeletons: {skeletons} → {new_skeletons} ({new_skeletons - skeletons:+d})")
    print(f"  Cows: {cows} → {new_cows} ({new_cows - cows:+d})")


def test_custom_config():
    """Test a custom configuration."""
    print(f"\n{'='*60}")
    print(f"Testing CUSTOM configuration")
    print(f"{'='*60}")
    
    # Create a custom config
    custom = {
        "coal_threshold": -0.5,
        "coal_probability": 0.5,
        "iron_threshold": -0.2,
        "iron_probability": 0.6,
        "diamond_threshold": 0.0,
        "diamond_probability": 0.1,
        "tree_threshold": -0.5,
        "tree_probability": 0.6,
        "cow_spawn_probability": 0.1,
        "cow_min_distance": 1,
        "zombie_spawn_probability": 0.0,
        "zombie_min_distance": 100,
        "skeleton_spawn_probability": 0.0
    }
    
    # Create environment
    env = Env(seed=42, world_config=custom)
    obs = env.reset()
    
    print("Custom config loaded successfully!")
    print(f"World shape: {obs.shape}")
    
    # Count resources in entire world
    world = env._world
    total_resources = {'coal': 0, 'iron': 0, 'diamond': 0, 'tree': 0}
    
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            mat_id = world._mat_map[x, y]
            for name, id_val in world._mat_ids.items():
                if id_val == mat_id and name in total_resources:
                    total_resources[name] += 1
    
    print(f"\nTotal resources in world:")
    for resource, count in total_resources.items():
        percentage = (count / (world.area[0] * world.area[1])) * 100
        print(f"  {resource}: {count} ({percentage:.1f}%)")


def main():
    print("Testing Custom Crafter Configurations")
    print("=" * 60)
    
    # Test all presets
    for config in ["easy", "normal", "hard", "peaceful"]:
        try:
            test_configuration(config)
        except Exception as e:
            print(f"Error testing {config}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test custom config
    try:
        test_custom_config()
    except Exception as e:
        print(f"Error testing custom config: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All tests complete!")
    print("\nSummary:")
    print("- EASY: More resources, fewer enemies")
    print("- NORMAL: Standard crafter experience")
    print("- HARD: Scarce resources, many enemies")
    print("- PEACEFUL: No enemies, abundant resources")
    print("- CUSTOM: Fully configurable via dict/JSON")


if __name__ == "__main__":
    main()