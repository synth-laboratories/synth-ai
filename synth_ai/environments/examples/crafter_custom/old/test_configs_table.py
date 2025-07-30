#!/usr/bin/env python3
"""
Test custom Crafter configurations with table output
===================================================
"""

import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from crafter import Env


def analyze_world(env):
    """Analyze world and return statistics."""
    world = env._world
    player = env._player
    
    # Count entities
    entities = {'Zombie': 0, 'Skeleton': 0, 'Cow': 0, 'Tree': 0}
    for obj in world._objects:
        if obj is None:
            continue
        obj_name = obj.__class__.__name__
        if obj_name in entities:
            entities[obj_name] += 1
    
    # Count resources in entire world
    resources = {'coal': 0, 'iron': 0, 'diamond': 0, 'tree': 0, 'lava': 0}
    total_tiles = world.area[0] * world.area[1]
    
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            mat_id = world._mat_map[x, y]
            for name, id_val in world._mat_ids.items():
                if id_val == mat_id and name in resources:
                    resources[name] += 1
    
    # Count resources near spawn (20x20 area)
    px, py = player.pos
    near_resources = {'coal': 0, 'iron': 0, 'diamond': 0}
    
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            x, y = px + dx, py + dy
            if 0 <= x < world.area[0] and 0 <= y < world.area[1]:
                mat_id = world._mat_map[x, y]
                for name, id_val in world._mat_ids.items():
                    if id_val == mat_id and name in near_resources:
                        near_resources[name] += 1
    
    return {
        'entities': entities,
        'resources': resources,
        'near_resources': near_resources,
        'total_tiles': total_tiles
    }


def test_configuration_analysis():
    """Test configuration analysis with different seeds."""
    print("Crafter Configuration Analysis")
    print("=" * 80)
    print("Testing with different seeds for world generation")
    print("=" * 80)
    
    seeds = [42, 123, 456, 789, 999]
    results = {}
    
    # Analyze each seed
    for seed in seeds:
        env = Env(seed=seed)
        env.reset()
        results[seed] = analyze_world(env)
    
    # Print entity comparison table
    print("\nINITIAL ENTITY COUNTS")
    print("-" * 60)
    print(f"{'Seed':<10} {'Zombies':<10} {'Skeletons':<12} {'Cows':<10} {'Trees':<10}")
    print("-" * 60)
    
    for seed in seeds:
        entities = results[seed]['entities']
        print(f"{seed:<10} {entities['Zombie']:<10} {entities['Skeleton']:<12} "
              f"{entities['Cow']:<10} {entities['Tree']:<10}")
    
    # Print resource comparison table (whole world)
    print("\nTOTAL RESOURCE BLOCKS IN WORLD (64x64 = 4096 tiles)")
    print("-" * 80)
    print(f"{'Seed':<10} {'Coal':<12} {'Iron':<12} {'Diamond':<12} {'Trees':<12} {'Lava':<12}")
    print(f"{'':10} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12}")
    print("-" * 80)
    
    for seed in seeds:
        res = results[seed]['resources']
        total = results[seed]['total_tiles']
        print(f"{seed:<10} ", end="")
        for resource in ['coal', 'iron', 'diamond', 'tree', 'lava']:
            count = res[resource]
            pct = (count / total) * 100
            print(f"{count:4d}/{pct:4.1f}%   ", end="")
        print()
    
    # Print resources near spawn
    print("\nRESOURCES NEAR SPAWN (20x20 area = 400 tiles)")
    print("-" * 60)
    print(f"{'Seed':<10} {'Coal':<15} {'Iron':<15} {'Diamond':<15}")
    print("-" * 60)
    
    for seed in seeds:
        near = results[seed]['near_resources']
        print(f"{seed:<10} {near['coal']:<15} {near['iron']:<15} {near['diamond']:<15}")
    
    # Test dynamic spawning
    print("\nDYNAMIC SPAWNING TEST (100 steps)")
    print("-" * 80)
    print(f"{'Seed':<10} {'Zombies':<20} {'Skeletons':<22} {'Cows':<20}")
    print(f"{'':10} {'(start → end)':20} {'(start → end)':22} {'(start → end)':20}")
    print("-" * 80)
    
    for seed in seeds:
        env = Env(seed=seed)
        env.reset()
        
        # Initial counts
        initial = analyze_world(env)
        
        # Take 100 steps
        for _ in range(100):
            # Use random action instead of action_space.sample()
            action = random.randint(0, env.action_space.n - 1)
            env.step(action)
        
        # Final counts
        final = analyze_world(env)
        
        # Print changes
        print(f"{seed:<10} ", end="")
        for entity in ['Zombie', 'Skeleton', 'Cow']:
            start = initial['entities'][entity]
            end = final['entities'][entity]
            diff = end - start
            print(f"{start:3d} → {end:3d} ({diff:+3d})     ", end="")
        print()
    
    # Print configuration details
    print("\nCONFIGURATION PARAMETERS")
    print("=" * 80)
    print("Note: WorldGenConfig not available in this version of crafter")
    print("Configuration parameters are handled internally by the Env class")


def main():
    """Run the configuration analysis."""
    test_configuration_analysis()


if __name__ == "__main__":
    main()