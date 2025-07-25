#!/usr/bin/env python3
"""
Test custom Crafter configurations with table output
===================================================
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from crafter import Env, WorldGenConfig


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


def main():
    print("Custom Crafter Configuration Analysis")
    print("=" * 80)
    print("Testing with seed=42 for all configurations")
    print("=" * 80)
    
    configs = ["easy", "normal", "hard", "peaceful", "resource_rich"]
    results = {}
    
    # Analyze each configuration
    for config in configs:
        env = Env(seed=42, world_config=config)
        env.reset()
        results[config] = analyze_world(env)
    
    # Print entity comparison table
    print("\nINITIAL ENTITY COUNTS")
    print("-" * 60)
    print(f"{'Config':<10} {'Zombies':<10} {'Skeletons':<12} {'Cows':<10} {'Trees':<10}")
    print("-" * 60)
    
    for config in configs:
        entities = results[config]['entities']
        print(f"{config:<10} {entities['Zombie']:<10} {entities['Skeleton']:<12} "
              f"{entities['Cow']:<10} {entities['Tree']:<10}")
    
    # Print resource comparison table (whole world)
    print("\nTOTAL RESOURCE BLOCKS IN WORLD (64x64 = 4096 tiles)")
    print("-" * 80)
    print(f"{'Config':<10} {'Coal':<12} {'Iron':<12} {'Diamond':<12} {'Trees':<12} {'Lava':<12}")
    print(f"{'':10} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12} {'(count/%)':12}")
    print("-" * 80)
    
    for config in configs:
        res = results[config]['resources']
        total = results[config]['total_tiles']
        print(f"{config:<10} ", end="")
        for resource in ['coal', 'iron', 'diamond', 'tree', 'lava']:
            count = res[resource]
            pct = (count / total) * 100
            print(f"{count:4d}/{pct:4.1f}%   ", end="")
        print()
    
    # Print resources near spawn
    print("\nRESOURCES NEAR SPAWN (20x20 area = 400 tiles)")
    print("-" * 60)
    print(f"{'Config':<10} {'Coal':<15} {'Iron':<15} {'Diamond':<15}")
    print("-" * 60)
    
    for config in configs:
        near = results[config]['near_resources']
        print(f"{config:<10} {near['coal']:<15} {near['iron']:<15} {near['diamond']:<15}")
    
    # Test dynamic spawning
    print("\nDYNAMIC SPAWNING TEST (100 steps)")
    print("-" * 80)
    print(f"{'Config':<10} {'Zombies':<20} {'Skeletons':<22} {'Cows':<20}")
    print(f"{'':10} {'(start → end)':20} {'(start → end)':22} {'(start → end)':20}")
    print("-" * 80)
    
    for config in configs:
        env = Env(seed=42, world_config=config)
        env.reset()
        
        # Initial counts
        initial = analyze_world(env)
        
        # Take 100 steps
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
        
        # Final counts
        final = analyze_world(env)
        
        # Print changes
        print(f"{config:<10} ", end="")
        for entity in ['Zombie', 'Skeleton', 'Cow']:
            start = initial['entities'][entity]
            end = final['entities'][entity]
            diff = end - start
            print(f"{start:3d} → {end:3d} ({diff:+3d})     ", end="")
        print()
    
    # Print configuration details
    print("\nCONFIGURATION PARAMETERS")
    print("=" * 80)
    
    # Load configs to show key parameters
    preset_configs = {
        'easy': WorldGenConfig.from_preset('easy'),
        'normal': WorldGenConfig.from_preset('normal'),
        'hard': WorldGenConfig.from_preset('hard'),
        'peaceful': WorldGenConfig.from_preset('peaceful'),
        'resource_rich': WorldGenConfig.from_preset('resource_rich')
    }
    
    params_to_show = [
        ('Resource Generation', [
            'coal_probability', 'iron_probability', 'diamond_probability', 'tree_probability'
        ]),
        ('Initial Spawning', [
            'cow_spawn_probability', 'zombie_spawn_probability', 'skeleton_spawn_probability'
        ]),
        ('Dynamic Spawning', [
            'zombie_max_count', 'skeleton_max_count', 'cow_max_count'
        ])
    ]
    
    for section, params in params_to_show:
        print(f"\n{section}:")
        print("-" * 60)
        
        # Print header
        print(f"{'Parameter':<25}", end="")
        for config in configs:
            print(f"{config:<12}", end="")
        print()
        print("-" * 60)
        
        # Print values
        for param in params:
            print(f"{param:<25}", end="")
            for config in configs:
                value = getattr(preset_configs[config], param)
                if isinstance(value, float):
                    print(f"{value:<12.3f}", end="")
                else:
                    print(f"{value:<12}", end="")
            print()


if __name__ == "__main__":
    main()