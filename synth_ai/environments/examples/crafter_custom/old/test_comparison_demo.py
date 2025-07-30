#!/usr/bin/env python3
"""
Demo comparison of resource_rich vs hard Crafter environments
Shows the dramatic difference in resource availability
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from crafter import Env
from collections import defaultdict
import numpy as np


def analyze_world(env):
    """Analyze world resources and entities."""
    world = env._world
    player = env._player
    
    # Count resources in entire world
    resources = defaultdict(int)
    total_tiles = world.area[0] * world.area[1]
    
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            mat_id = world._mat_map[x, y]
            for name, id_val in world._mat_ids.items():
                if id_val == mat_id:
                    resources[name] += 1
    
    # Count entities
    entities = defaultdict(int)
    for obj in world._objects:
        if obj is None:
            continue
        entities[obj.__class__.__name__] += 1
    
    # Count resources near spawn (20x20 area)
    px, py = player.pos
    near_resources = defaultdict(int)
    
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            x, y = px + dx, py + dy
            if 0 <= x < world.area[0] and 0 <= y < world.area[1]:
                mat_id = world._mat_map[x, y]
                for name, id_val in world._mat_ids.items():
                    if id_val == mat_id:
                        near_resources[name] += 1
    
    return {
        'resources': dict(resources),
        'entities': dict(entities),
        'near_resources': dict(near_resources),
        'total_tiles': total_tiles
    }


def simulate_agent_run(difficulty, seed, steps=100):
    """Simulate a simple agent run to show achievable progress."""
    env = Env(seed=seed, world_config=difficulty)
    obs = env.reset()
    
    achievements = set()
    inventory = {}
    
    # Simple agent that moves randomly but prefers to "do" when near resources
    for _ in range(steps):
        # Check what's around us using semantic view
        sem_map = env._sem_view()
        px, py = env._player.pos
        
        # Check 3x3 area around player
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = px + dx, py + dy
                if 0 <= x < sem_map.shape[0] and 0 <= y < sem_map.shape[1]:
                    nearby.append(sem_map[x, y])
        
        # If there's something interesting nearby, do action
        # IDs: tree=7, coal=4, iron=5, diamond=6, cow=14
        interesting = [4, 5, 6, 7, 14]
        if any(item in nearby for item in interesting):
            action = 5  # do
        else:
            # Random movement
            action = np.random.choice([1, 2, 3, 4])  # move
        
        obs, reward, done, info = env.step(action)
        
        # Track achievements
        if 'achievements' in info:
            for ach, status in info['achievements'].items():
                if status:
                    achievements.add(ach)
        
        # Track inventory
        if 'inventory' in info:
            inventory = info['inventory']
        
        if done:
            break
    
    return {
        'achievements': list(achievements),
        'inventory': inventory,
        'final_health': env._player.health
    }


def main():
    print("🎮 Crafter Custom Environment Comparison")
    print("Comparing resource_rich vs hard difficulties")
    print("=" * 80)
    
    # Test multiple seeds
    seeds = [42, 123, 456, 789, 1000]
    
    # Analyze world generation
    print("\n📊 WORLD GENERATION ANALYSIS (5 seeds each)")
    print("=" * 80)
    
    rr_stats = {'coal': [], 'iron': [], 'diamond': [], 'tree': [], 'Zombie': [], 'Skeleton': []}
    hard_stats = {'coal': [], 'iron': [], 'diamond': [], 'tree': [], 'Zombie': [], 'Skeleton': []}
    
    for seed in seeds:
        # Resource rich
        env_rr = Env(seed=seed, world_config="resource_rich")
        env_rr.reset()
        analysis_rr = analyze_world(env_rr)
        
        # Hard
        env_hard = Env(seed=seed, world_config="hard")
        env_hard.reset()
        analysis_hard = analyze_world(env_hard)
        
        # Collect stats
        for resource in ['coal', 'iron', 'diamond', 'tree']:
            rr_stats[resource].append(analysis_rr['resources'].get(resource, 0))
            hard_stats[resource].append(analysis_hard['resources'].get(resource, 0))
        
        for entity in ['Zombie', 'Skeleton']:
            rr_stats[entity].append(analysis_rr['entities'].get(entity, 0))
            hard_stats[entity].append(analysis_hard['entities'].get(entity, 0))
    
    # Print comparison
    print(f"{'Resource/Entity':<20} {'Resource Rich (avg)':<25} {'Hard (avg)':<25} {'Ratio':<10}")
    print("-" * 80)
    
    for key in ['coal', 'iron', 'diamond', 'tree', 'Zombie', 'Skeleton']:
        rr_avg = np.mean(rr_stats[key])
        hard_avg = np.mean(hard_stats[key])
        ratio = rr_avg / hard_avg if hard_avg > 0 else float('inf')
        
        print(f"{key:<20} {rr_avg:<25.1f} {hard_avg:<25.1f} {ratio:<10.1f}x")
    
    # Simulate agent runs
    print("\n🤖 SIMULATED AGENT PERFORMANCE (100 steps, 5 runs each)")
    print("=" * 80)
    
    rr_achievements = []
    hard_achievements = []
    
    for seed in seeds:
        rr_result = simulate_agent_run("resource_rich", seed)
        hard_result = simulate_agent_run("hard", seed)
        
        rr_achievements.extend(rr_result['achievements'])
        hard_achievements.extend(hard_result['achievements'])
    
    # Count unique achievements
    rr_unique = set(rr_achievements)
    hard_unique = set(hard_achievements)
    
    print(f"\nUnique achievements unlocked:")
    print(f"Resource Rich: {len(rr_unique)} - {sorted(rr_unique)}")
    print(f"Hard: {len(hard_unique)} - {sorted(hard_unique)}")
    
    # Achievement frequency
    print(f"\nAchievement frequency (out of 5 runs):")
    achievement_names = sorted(rr_unique | hard_unique)
    
    if achievement_names:
        print(f"{'Achievement':<25} {'Resource Rich':<20} {'Hard':<20}")
        print("-" * 65)
        
        for ach in achievement_names:
            rr_count = rr_achievements.count(ach)
            hard_count = hard_achievements.count(ach)
            print(f"{ach:<25} {rr_count}/5{'':<16} {hard_count}/5")
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    print("-" * 40)
    
    # Resource abundance
    diamond_ratio = np.mean(rr_stats['diamond']) / (np.mean(hard_stats['diamond']) + 0.001)
    iron_ratio = np.mean(rr_stats['iron']) / (np.mean(hard_stats['iron']) + 0.001)
    
    print(f"✅ Diamonds are {diamond_ratio:.0f}x more common in resource_rich")
    print(f"✅ Iron is {iron_ratio:.0f}x more common in resource_rich")
    
    # Enemy density
    enemy_rr = np.mean(rr_stats['Zombie']) + np.mean(rr_stats['Skeleton'])
    enemy_hard = np.mean(hard_stats['Zombie']) + np.mean(hard_stats['Skeleton'])
    
    print(f"✅ Hard mode has {enemy_hard:.0f} enemies vs {enemy_rr:.0f} in resource_rich")
    
    # Achievement difference
    if len(rr_unique) > len(hard_unique):
        print(f"✅ Resource rich unlocked {len(rr_unique) - len(hard_unique)} more achievement types")
    
    print("\n💡 CONCLUSION:")
    print("The resource_rich configuration provides dramatically more resources,")
    print("especially diamonds and iron, making advanced crafting much more achievable.")
    print("Meanwhile, hard mode is extremely challenging with scarce resources and many enemies.")


if __name__ == "__main__":
    main()