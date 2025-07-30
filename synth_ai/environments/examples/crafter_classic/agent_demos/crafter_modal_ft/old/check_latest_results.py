#!/usr/bin/env python3
"""Check the latest results JSON file."""

import json
import glob
from pathlib import Path

# Find the latest results file
result_files = glob.glob("crafter_lm_synth_results_*.json")
if not result_files:
    print("No result files found")
    exit(1)

# Get the most recent file
latest_file = max(result_files, key=lambda f: Path(f).stat().st_mtime)
print(f"📊 Checking latest results: {latest_file}\n")

with open(latest_file) as f:
    data = json.load(f)

# Extract key metrics
total_episodes = data.get('total_episodes', 0)
total_steps = data.get('total_steps', 0)
model = data.get('model', 'unknown')

print(f"Model: {model}")
print(f"Episodes: {total_episodes}")
print(f"Total Steps: {total_steps}")

# Check episode results
episodes = data.get('episodes', [])
if episodes:
    print(f"\n📊 Episode Summary:")
    for i, ep in enumerate(episodes):
        steps = ep.get('steps', 0)
        reward = ep.get('total_reward', 0)
        achievements = ep.get('achievements_unlocked', [])
        
        print(f"\nEpisode {i}:")
        print(f"  Steps: {steps}")
        print(f"  Reward: {reward}")
        print(f"  Achievements: {len(achievements)}")
        if achievements:
            print(f"    - {', '.join(achievements[:5])}")
            if len(achievements) > 5:
                print(f"    ... and {len(achievements) - 5} more")
        
        # Check inventory at end
        inventory = ep.get('final_inventory', {})
        non_zero = {k: v for k, v in inventory.items() if v > 0 and k not in ['health', 'food', 'drink', 'energy']}
        if non_zero:
            print(f"  Final inventory: {non_zero}")
        
        # Sample actions
        if 'action_history' in ep and ep['action_history']:
            print(f"  Sample actions: {ep['action_history'][:5]}")

# Overall statistics
avg_reward = sum(ep.get('total_reward', 0) for ep in episodes) / len(episodes) if episodes else 0
total_achievements = sum(len(ep.get('achievements_unlocked', [])) for ep in episodes)

print(f"\n📊 Overall Statistics:")
print(f"Average reward per episode: {avg_reward:.2f}")
print(f"Total achievements unlocked: {total_achievements}")