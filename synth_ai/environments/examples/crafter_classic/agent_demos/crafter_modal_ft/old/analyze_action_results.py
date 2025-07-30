#!/usr/bin/env python3
"""Analyze why actions aren't producing expected results."""

import json
import glob
from pathlib import Path

# Find the latest results file
result_files = glob.glob("crafter_lm_synth_results_*.json")
if not result_files:
    print("No result files found")
    exit(1)

latest_file = max(result_files, key=lambda f: Path(f).stat().st_mtime)
print(f"📊 Analyzing: {latest_file}\n")

with open(latest_file) as f:
    data = json.load(f)

# Check each episode
for ep_idx, episode in enumerate(data.get('results', [])):
    if 'error' in episode:
        continue
        
    print(f"\n{'='*60}")
    print(f"EPISODE {ep_idx}")
    print(f"{'='*60}")
    
    steps = episode.get('step_results', [])
    
    # Track inventory changes
    prev_inventory = {}
    
    for step in steps:
        turn = step.get('turn', 0)
        action = step.get('action', 'unknown')
        reward = step.get('reward', 0)
        
        # This won't work because we don't store obs in step_results
        # We need to look at the actual traces or add obs to step_results
        
    achievements = episode.get('achievements_unlocked', [])
    print(f"\nFinal achievements: {achievements}")
    print(f"Total reward: {episode.get('total_reward', 0)}")
    print(f"Steps: {episode.get('steps', 0)}")

# Summary
summary = data.get('summary', {})
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"Unique achievements: {summary.get('unique_achievements', [])}")
print(f"Average reward: {summary.get('avg_reward', 0)}")