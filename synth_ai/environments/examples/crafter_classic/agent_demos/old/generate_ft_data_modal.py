#!/usr/bin/env python3
"""
Script to generate and filter fine-tuning data from Crafter rollouts
Applies quality filters and formats data for Modal fine-tuning
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CrafterDataFilter:
    """Filter and process Crafter rollout data for fine-tuning."""
    
    def __init__(
        self,
        min_achievements: int = 3,
        min_reward: float = 0.0,
        max_steps: int = 100,
        min_steps: int = 10,
        achievement_weights: Optional[Dict[str, float]] = None
    ):
        self.min_achievements = min_achievements
        self.min_reward = min_reward
        self.max_steps = max_steps
        self.min_steps = min_steps
        
        # Default achievement weights (higher = more valuable)
        self.achievement_weights = achievement_weights or {
            # Basic resources
            "collect_wood": 1.0,
            "collect_stone": 1.5,
            "collect_coal": 3.0,
            "collect_iron": 5.0,
            "collect_diamond": 10.0,
            
            # Crafting
            "place_table": 2.0,
            "place_furnace": 3.0,
            "make_wood_pickaxe": 2.5,
            "make_stone_pickaxe": 3.5,
            "make_iron_pickaxe": 6.0,
            "make_wood_sword": 2.5,
            "make_stone_sword": 3.5,
            "make_iron_sword": 6.0,
            
            # Survival
            "eat_cow": 2.0,
            "eat_plant": 1.0,
            "collect_drink": 1.0,
            "sleep": 1.5,
            
            # Combat
            "defeat_zombie": 3.0,
            "defeat_skeleton": 4.0,
        }
    
    def calculate_episode_score(self, episode: dict) -> float:
        """Calculate a quality score for an episode."""
        score = 0.0
        
        # Achievement score
        achievements = episode.get("achievements", [])
        for ach in achievements:
            score += self.achievement_weights.get(ach, 1.0)
        
        # Efficiency bonus (achievements per step)
        num_steps = len(episode.get("steps", []))
        if num_steps > 0:
            efficiency = len(achievements) / num_steps
            score += efficiency * 10
        
        # Reward contribution
        total_reward = episode.get("total_reward", 0)
        score += total_reward * 0.5
        
        # Diversity bonus (unique actions)
        unique_actions = len(set(step["action"] for step in episode.get("steps", [])))
        score += unique_actions * 0.5
        
        return score
    
    def filter_episode(self, episode: dict) -> Tuple[bool, str]:
        """
        Check if an episode passes quality filters.
        Returns (passes, reason).
        """
        # Check achievements
        achievements = episode.get("achievements", [])
        if len(achievements) < self.min_achievements:
            return False, f"Too few achievements: {len(achievements)} < {self.min_achievements}"
        
        # Check reward
        total_reward = episode.get("total_reward", 0)
        if total_reward < self.min_reward:
            return False, f"Low reward: {total_reward} < {self.min_reward}"
        
        # Check step count
        num_steps = len(episode.get("steps", []))
        if num_steps < self.min_steps:
            return False, f"Too few steps: {num_steps} < {self.min_steps}"
        if num_steps > self.max_steps:
            return False, f"Too many steps: {num_steps} > {self.max_steps}"
        
        # Check for errors
        if episode.get("termination_reason", "").startswith("error"):
            return False, "Episode ended with error"
        
        return True, "Passed all filters"
    
    def optimize_conversation(self, messages: List[dict]) -> List[dict]:
        """
        Optimize conversation for fine-tuning by:
        - Removing redundant information
        - Condensing observations
        - Ensuring proper format
        """
        optimized = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Keep system message as-is
                optimized.append(msg)
            
            elif role == "user":
                # Condense user observations
                if len(content) > 1000:
                    # Extract key information
                    lines = content.split("\n")
                    key_lines = []
                    
                    for line in lines:
                        # Keep important lines
                        if any(keyword in line.lower() for keyword in [
                            "health:", "hunger:", "thirst:", "inventory:",
                            "achievements:", "map", "recent"
                        ]):
                            key_lines.append(line)
                    
                    # Keep reasoning prompt
                    if "Think step by step:" in content:
                        idx = content.index("Think step by step:")
                        key_lines.append(content[idx:])
                    
                    content = "\n".join(key_lines)
                
                optimized.append({"role": role, "content": content})
            
            elif role == "assistant":
                # Keep assistant responses but trim if very long
                if len(content) > 500:
                    # Keep first part (reasoning) and action
                    lines = content.split("\n")
                    kept_lines = []
                    
                    # Keep reasoning (usually first few lines)
                    for i, line in enumerate(lines[:5]):
                        kept_lines.append(line)
                    
                    # Find and keep action
                    for line in lines:
                        if any(action in line for action in [
                            "move_", "do", "sleep", "place_", "make_"
                        ]):
                            if line not in kept_lines:
                                kept_lines.append(line)
                    
                    content = "\n".join(kept_lines)
                
                optimized.append({"role": role, "content": content})
        
        return optimized
    
    def create_training_example(self, episode: dict) -> dict:
        """Create a training example from a filtered episode."""
        messages = episode.get("messages", [])
        
        # Optimize conversation
        optimized_messages = self.optimize_conversation(messages)
        
        # Add metadata as comment in system message
        metadata_comment = (
            f"# Episode stats: {len(episode['achievements'])} achievements, "
            f"{episode['total_reward']:.1f} reward, {len(episode['steps'])} steps"
        )
        
        if optimized_messages and optimized_messages[0]["role"] == "system":
            optimized_messages[0]["content"] += f"\n{metadata_comment}"
        
        return {"messages": optimized_messages}


def analyze_rollout_directory(rollout_dir: Path) -> Dict[str, any]:
    """Analyze a rollout directory and return statistics."""
    stats = {
        "total_episodes": 0,
        "total_achievements": defaultdict(int),
        "reward_distribution": [],
        "step_distribution": [],
        "achievement_distribution": [],
    }
    
    # Find rollout data files
    rollout_files = list(rollout_dir.glob("**/rollout_data.json"))
    
    for rollout_file in rollout_files:
        with open(rollout_file, "r") as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        stats["total_episodes"] += len(episodes)
        
        for ep in episodes:
            # Track achievements
            for ach in ep.get("achievements", []):
                stats["total_achievements"][ach] += 1
            
            # Track distributions
            stats["reward_distribution"].append(ep.get("total_reward", 0))
            stats["step_distribution"].append(len(ep.get("steps", [])))
            stats["achievement_distribution"].append(len(ep.get("achievements", [])))
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning data from Crafter rollouts")
    parser.add_argument("rollout_dir", type=str, help="Directory containing rollout data")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: rollout_dir/fine_tuning_filtered.jsonl)")
    parser.add_argument("--min-achievements", type=int, default=3,
                       help="Minimum achievements required")
    parser.add_argument("--min-reward", type=float, default=0.0,
                       help="Minimum total reward required")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps allowed")
    parser.add_argument("--min-steps", type=int, default=10,
                       help="Minimum steps required")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Only keep top K episodes by score")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze data, don't generate output")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed filtering information")
    
    args = parser.parse_args()
    
    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        print(f"❌ Rollout directory not found: {rollout_dir}")
        return 1
    
    # Analyze rollout data
    print(f"📊 Analyzing rollout data in {rollout_dir}...")
    stats = analyze_rollout_directory(rollout_dir)
    
    print(f"\n📈 Rollout Statistics:")
    print(f"   Total episodes: {stats['total_episodes']}")
    
    if stats['total_episodes'] == 0:
        print("❌ No episodes found in rollout directory")
        return 1
    
    print(f"   Avg achievements: {np.mean(stats['achievement_distribution']):.1f}")
    print(f"   Avg reward: {np.mean(stats['reward_distribution']):.1f}")
    print(f"   Avg steps: {np.mean(stats['step_distribution']):.1f}")
    
    print(f"\n🏆 Top achievements:")
    for ach, count in sorted(stats['total_achievements'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]:
        pct = count / stats['total_episodes'] * 100
        print(f"   {ach}: {count} ({pct:.1f}%)")
    
    if args.analyze_only:
        return 0
    
    # Create filter
    filter_config = CrafterDataFilter(
        min_achievements=args.min_achievements,
        min_reward=args.min_reward,
        max_steps=args.max_steps,
        min_steps=args.min_steps
    )
    
    # Process episodes
    print(f"\n🔍 Filtering episodes...")
    all_episodes = []
    filtered_episodes = []
    filter_reasons = defaultdict(int)
    
    # Load all episodes
    rollout_files = list(rollout_dir.glob("**/rollout_data.json"))
    for rollout_file in rollout_files:
        with open(rollout_file, "r") as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        all_episodes.extend(episodes)
    
    # Filter episodes
    for ep in all_episodes:
        passes, reason = filter_config.filter_episode(ep)
        if passes:
            # Calculate score for ranking
            score = filter_config.calculate_episode_score(ep)
            filtered_episodes.append((score, ep))
        else:
            filter_reasons[reason] += 1
            if args.verbose:
                print(f"   Filtered out: {reason}")
    
    print(f"\n📋 Filtering Results:")
    print(f"   Total episodes: {len(all_episodes)}")
    print(f"   Passed filters: {len(filtered_episodes)}")
    print(f"   Pass rate: {len(filtered_episodes)/len(all_episodes)*100:.1f}%")
    
    if filter_reasons:
        print(f"\n❌ Filter reasons:")
        for reason, count in sorted(filter_reasons.items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"   {reason}: {count}")
    
    if not filtered_episodes:
        print("\n❌ No episodes passed the filters!")
        return 1
    
    # Sort by score and apply top-k if specified
    filtered_episodes.sort(key=lambda x: x[0], reverse=True)
    
    if args.top_k and args.top_k < len(filtered_episodes):
        print(f"\n🎯 Selecting top {args.top_k} episodes by score")
        filtered_episodes = filtered_episodes[:args.top_k]
    
    # Generate training examples
    print(f"\n✍️  Generating training examples...")
    training_examples = []
    
    for score, episode in filtered_episodes:
        example = filter_config.create_training_example(episode)
        training_examples.append(example)
        
        if args.verbose:
            achievements = episode.get("achievements", [])
            print(f"   Score: {score:.1f}, Achievements: {achievements}")
    
    # Save output
    output_file = args.output
    if not output_file:
        output_file = rollout_dir / "fine_tuning_filtered.jsonl"
    else:
        output_file = Path(output_file)
    
    with open(output_file, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"\n✅ Generated {len(training_examples)} training examples")
    print(f"📁 Saved to: {output_file}")
    
    # Calculate token estimate
    total_chars = sum(len(json.dumps(ex)) for ex in training_examples)
    est_tokens = total_chars // 4
    print(f"📊 Estimated tokens: {est_tokens:,}")
    
    # Show sample
    print(f"\n📝 Sample training example:")
    sample = training_examples[0]["messages"]
    for msg in sample[:3]:  # Show first 3 messages
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   [{role}] {content}")
    
    return 0


if __name__ == "__main__":
    exit(main())