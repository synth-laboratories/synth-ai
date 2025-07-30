#!/usr/bin/env python3
"""
Test script to run ReAct agents against Custom Crafter environment with dataset instances
Tests on resource_rich vs hard difficulty instances
"""

import asyncio
import json
import uuid
import math
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from httpx import AsyncClient
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Disable Langfuse
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = "dummy"
os.environ["LANGFUSE_SECRET_KEY"] = "dummy"

from langfuse.openai import openai
import numpy as np

from synth_ai.environments.examples.crafter_custom.run_dataset import CrafterDatasetRunner
from synth_ai.environments.examples.crafter_custom.dataset_builder import CrafterDatasetBuilder


# --- Tool Definitions ---
def get_openai_tools():
    """Get OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform 1-5 actions in sequence in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 1-5 action names to execute in sequence"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why these actions were chosen"
                        }
                    },
                    "required": ["actions", "reasoning"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "End the episode when finished or no progress can be made.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for termination"
                        }
                    },
                    "required": ["reason"]
                }
            }
        }
    ]


def format_semantic_map_view(obs_data: Dict[str, Any], view_size: int = 7) -> str:
    """Format a semantic map view around the player (ASCII)."""
    try:
        # Simple ID to item mapping for Crafter
        id_to_item = [
            "void", "grass", "stone", "path", "coal", "iron", "diamond", 
            "tree", "table", "furnace", "water", "sand", "lava", "plant",
            "player", "cow", "zombie", "skeleton"
        ]
        
        semantic_map = obs_data.get("semantic_map")
        player_position = obs_data.get("player_position", [0, 0])
        
        if semantic_map is None:
            return "Map view unavailable"
        
        # Ensure numpy array
        sem_arr = np.asarray(semantic_map)
        if sem_arr.ndim == 1:
            size = int(np.sqrt(sem_arr.size))
            sem_arr = sem_arr.reshape(size, size)
        
        px, py = map(int, player_position)
        half = view_size // 2
        
        rows = []
        visible = set()
        for dy in range(-half, half + 1):
            row_tokens = []
            for dx in range(-half, half + 1):
                x, y = px + dx, py + dy
                if 0 <= x < sem_arr.shape[0] and 0 <= y < sem_arr.shape[1]:
                    if dx == 0 and dy == 0:
                        token = "player"
                    else:
                        idx = int(sem_arr[x, y])
                        token = id_to_item[idx] if idx < len(id_to_item) else "?"
                else:
                    token = "void"
                row_tokens.append(token)
                if token not in {"void", "player", "grass", "path", "water", "sand"}:
                    visible.add(token)
            rows.append(" ".join(row_tokens))
        
        map_view = f"\nLocal Map View ({view_size}x{view_size}):\n" + "\n".join(rows)
        if visible:
            map_view += "\nVisible items: " + ", ".join(sorted(visible))
        return map_view
    except Exception as e:
        return f"Map view error: {e}"


class CrafterReActAgent:
    """ReAct agent for Crafter environment."""
    
    def __init__(self, model_name: str, max_turns: int = 20, verbose: bool = False):
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.tools = get_openai_tools()
    
    def get_system_message(self, difficulty: str = None) -> str:
        """Get system message with optional difficulty-specific hints."""
        base_message = """You are playing Crafter. Your goal is to unlock as many achievements as possible.

Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop

Key mechanics:
• 'do' action: collect resources (wood, stone, coal, iron, diamond) and attack enemies
• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools
• Place table to enable crafting
• Sleep when energy low
• Eat food when health low

Use the semantic map to navigate toward resources."""
        
        if difficulty == "resource_rich":
            base_message += "\n\nThis is a RESOURCE RICH world - diamonds, iron, and coal are extremely abundant! Look for them everywhere."
        elif difficulty == "hard":
            base_message += "\n\nThis is a HARD world - resources are scarce and enemies are plentiful. Be very careful and efficient."
        
        return base_message
    
    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for the agent."""
        health = obs.get("health", 0)
        inventory = obs.get("inventory", {})
        
        # Extract health from inventory if needed
        if health == 0 and "health" in inventory:
            health = inventory["health"]
        
        # Format inventory
        inventory_items = []
        for item, count in inventory.items():
            if count > 0 and item != "health":
                inventory_items.append(f"{item}: {count}")
        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"
        
        # Get achievements
        achievements = obs.get("achievements") or obs.get("achievements_status", {})
        unlocked = [name for name, status in achievements.items() if status]
        achievements_str = ", ".join(unlocked) if unlocked else "none"
        
        # Get position
        position = obs.get("player_position", obs.get("position", [0, 0]))
        num_steps = obs.get("num_steps_taken", 0)
        
        # Get semantic map view
        map_view = format_semantic_map_view(obs, view_size=7)
        
        return (
            f"Step: {num_steps}\n"
            f"Health: {health}\n"
            f"Position: {position}\n"
            f"Inventory: {inventory_str}\n"
            f"Achievements: {achievements_str}\n"
            f"{map_view}\n\n"
            f"Choose 1-5 actions to execute. Focus on collecting resources and crafting."
        )
    
    def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision based on observation."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Turn {turn + 1}/{self.max_turns}\n\n{obs}"}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            tool_calls = response.choices[0].message.tool_calls
            
            if not tool_calls:
                return {
                    "name": "interact",
                    "parameters": {
                        "actions": ["do"],
                        "reasoning": "Default action"
                    }
                }
            
            tool_call = tool_calls[0]
            return {
                "name": tool_call.function.name,
                "parameters": json.loads(tool_call.function.arguments)
            }
            
        except Exception as e:
            print(f"Error in decide: {e}")
            return {
                "name": "interact",
                "parameters": {
                    "actions": ["noop"],
                    "reasoning": f"Error: {str(e)}"
                }
            }


async def run_single_custom_instance(
    client: AsyncClient,
    agent: CrafterReActAgent,
    instance: Dict[str, Any],
    instance_num: int
) -> Dict[str, Any]:
    """Run a single custom Crafter instance."""
    
    difficulty = instance["metadata"]["difficulty"]
    seed = instance["metadata"]["world_seed"]
    
    print(f"\n{'='*60}")
    print(f"Instance {instance_num}: {instance['id']}")
    print(f"Difficulty: {difficulty}, Seed: {seed}")
    print(f"Task: {instance['impetus']['instructions']}")
    
    try:
        # Create environment
        from crafter import Env
        env = Env(seed=seed)
        
        # We need to wrap this in the service API format
        # For now, we'll run it directly
        obs_array = env.reset()
        
        # Convert numpy observation to dict format
        obs = {
            "observation_image": obs_array.tolist() if hasattr(obs_array, 'tolist') else obs_array,
            "health": 9,  # Default starting health
            "inventory": {},
            "achievements_status": {},
            "player_position": [32, 32],  # Default center position
            "num_steps_taken": 0,
            "terminated": False
        }
        
        # Add semantic map if available
        if hasattr(env, '_sem_view'):
            obs["semantic_map"] = env._sem_view().tolist()
        
        total_reward = 0
        achievements_unlocked = set()
        
        # Run episode
        for turn in range(agent.max_turns):
            # Format observation
            formatted_obs = agent.format_observation(obs)
            
            # Get agent decision
            action = agent.decide(
                formatted_obs,
                agent.get_system_message(difficulty),
                turn
            )
            
            if action["name"] == "terminate":
                print(f"Agent terminated: {action['parameters']['reason']}")
                break
            
            # Execute actions
            action_sequence = action["parameters"]["actions"]
            
            # Action mapping
            ACTION_MAP = {
                "noop": 0, "move_left": 1, "move_right": 2, "move_up": 3,
                "move_down": 4, "do": 5, "sleep": 6, "place_stone": 7,
                "place_table": 8, "place_furnace": 9, "place_plant": 10,
                "make_wood_pickaxe": 11, "make_stone_pickaxe": 12,
                "make_iron_pickaxe": 13, "make_wood_sword": 14,
                "make_stone_sword": 15, "make_iron_sword": 16
            }
            
            # Execute each action
            for action_name in action_sequence:
                action_int = ACTION_MAP.get(action_name, 0)
                obs_array, reward, done, info = env.step(action_int)
                total_reward += reward
                
                # Update observation dict
                obs["observation_image"] = obs_array.tolist() if hasattr(obs_array, 'tolist') else obs_array
                obs["num_steps_taken"] = turn + 1
                obs["terminated"] = done
                
                # Update from info
                if "inventory" in info:
                    obs["inventory"] = info["inventory"]
                if "achievements" in info:
                    obs["achievements_status"] = info["achievements"]
                    # Track new achievements
                    for ach, status in info["achievements"].items():
                        if status and ach not in achievements_unlocked:
                            achievements_unlocked.add(ach)
                            print(f"  🏆 Achievement unlocked: {ach}")
                
                # Update semantic map
                if hasattr(env, '_sem_view'):
                    obs["semantic_map"] = env._sem_view().tolist()
                
                # Update player position
                if hasattr(env, '_player'):
                    obs["player_position"] = env._player.pos.tolist()
                
                if done:
                    break
            
            if obs["terminated"]:
                break
        
        # Results
        print(f"\nResults:")
        print(f"  Total reward: {total_reward}")
        print(f"  Achievements: {len(achievements_unlocked)} - {list(achievements_unlocked)}")
        
        return {
            "instance_id": instance["id"],
            "difficulty": difficulty,
            "seed": seed,
            "total_reward": total_reward,
            "achievements": list(achievements_unlocked),
            "num_achievements": len(achievements_unlocked),
            "steps": obs["num_steps_taken"]
        }
        
    except Exception as e:
        print(f"Error running instance: {e}")
        import traceback
        traceback.print_exc()
        return {
            "instance_id": instance["id"],
            "difficulty": difficulty,
            "seed": seed,
            "total_reward": 0,
            "achievements": [],
            "num_achievements": 0,
            "steps": 0,
            "error": str(e)
        }


async def main():
    """Run comparison between resource_rich and hard difficulties."""
    parser = argparse.ArgumentParser(description="Test Custom Crafter with OpenAI models")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")
    parser.add_argument("--max-turns", "-t", type=int, default=20, help="Max turns per episode")
    parser.add_argument("--episodes", "-e", type=int, default=5, help="Episodes per difficulty")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(logging.WARNING)
    
    print(f"🎮 Custom Crafter Agent Evaluation")
    print(f"Model: {args.model}")
    print(f"Episodes per difficulty: {args.episodes}")
    print(f"Max turns: {args.max_turns}")
    print("=" * 60)
    
    # Check if datasets exist, create if not
    dataset_path = Path("dataset")
    if not (dataset_path / "crafter_comparison_v1").exists():
        print("Creating comparison dataset...")
        builder = CrafterDatasetBuilder()
        dataset = builder.build_dataset(
            name="crafter_comparison_v1",
            instances_per_difficulty={
                "resource_rich": args.episodes,
                "hard": args.episodes
            }
        )
        builder.save_dataset(dataset)
    
    # Load dataset
    runner = CrafterDatasetRunner()
    
    # Test service endpoint (dummy client for now)
    async with AsyncClient(base_url="http://localhost:8901", timeout=30.0) as client:
        
        # Run resource_rich instances
        print(f"\n🌟 Testing RESOURCE RICH difficulty ({args.episodes} episodes)")
        print("-" * 60)
        
        dataset = runner.load_dataset("crafter_comparison_v1")
        resource_rich_instances = [
            inst for inst in dataset["instances"] 
            if inst["metadata"]["difficulty"] == "resource_rich"
        ][:args.episodes]
        
        resource_rich_results = []
        for i, instance in enumerate(resource_rich_instances, 1):
            agent = CrafterReActAgent(args.model, max_turns=args.max_turns)
            result = await run_single_custom_instance(client, agent, instance, i)
            resource_rich_results.append(result)
        
        # Run hard instances
        print(f"\n💀 Testing HARD difficulty ({args.episodes} episodes)")
        print("-" * 60)
        
        hard_instances = [
            inst for inst in dataset["instances"] 
            if inst["metadata"]["difficulty"] == "hard"
        ][:args.episodes]
        
        hard_results = []
        for i, instance in enumerate(hard_instances, 1):
            agent = CrafterReActAgent(args.model, max_turns=args.max_turns)
            result = await run_single_custom_instance(client, agent, instance, i)
            hard_results.append(result)
        
        # Print comparison
        print("\n" + "=" * 80)
        print("🏆 COMPARISON RESULTS")
        print("=" * 80)
        
        # Calculate statistics
        def calc_stats(results):
            achievements_all = defaultdict(int)
            for r in results:
                for ach in r["achievements"]:
                    achievements_all[ach] += 1
            
            return {
                "avg_achievements": sum(r["num_achievements"] for r in results) / len(results),
                "avg_reward": sum(r["total_reward"] for r in results) / len(results),
                "unique_achievements": len(set(ach for r in results for ach in r["achievements"])),
                "achievement_counts": dict(achievements_all)
            }
        
        rr_stats = calc_stats(resource_rich_results)
        hard_stats = calc_stats(hard_results)
        
        # Print comparison table
        print(f"{'Metric':<30} {'Resource Rich':<20} {'Hard':<20} {'Difference':<20}")
        print("-" * 90)
        
        avg_ach_diff = rr_stats["avg_achievements"] - hard_stats["avg_achievements"]
        print(f"{'Avg Achievements/Episode':<30} {rr_stats['avg_achievements']:<20.1f} {hard_stats['avg_achievements']:<20.1f} {avg_ach_diff:+.1f}")
        
        unique_diff = rr_stats["unique_achievements"] - hard_stats["unique_achievements"]
        print(f"{'Unique Achievements Total':<30} {rr_stats['unique_achievements']:<20} {hard_stats['unique_achievements']:<20} {unique_diff:+d}")
        
        reward_diff = rr_stats["avg_reward"] - hard_stats["avg_reward"]
        print(f"{'Avg Reward/Episode':<30} {rr_stats['avg_reward']:<20.1f} {hard_stats['avg_reward']:<20.1f} {reward_diff:+.1f}")
        
        # Achievement breakdown
        print("\n📊 Achievement Breakdown:")
        print(f"{'Achievement':<25} {'Resource Rich':<15} {'Hard':<15}")
        print("-" * 55)
        
        all_achievements = set(rr_stats["achievement_counts"].keys()) | set(hard_stats["achievement_counts"].keys())
        for ach in sorted(all_achievements):
            rr_count = rr_stats["achievement_counts"].get(ach, 0)
            hard_count = hard_stats["achievement_counts"].get(ach, 0)
            print(f"{ach:<25} {rr_count:<15} {hard_count:<15}")
        
        # Key findings
        print("\n🔍 Key Findings:")
        if rr_stats["avg_achievements"] > hard_stats["avg_achievements"] * 1.5:
            print("✅ Resource rich environment yields significantly more achievements")
        elif rr_stats["avg_achievements"] > hard_stats["avg_achievements"]:
            print("✅ Resource rich environment yields moderately more achievements")
        else:
            print("❌ No significant advantage in resource rich environment")
        
        # Check specific resource-dependent achievements
        resource_achievements = ["collect_diamond", "collect_iron", "make_iron_pickaxe", "make_iron_sword"]
        rr_resource = sum(rr_stats["achievement_counts"].get(ach, 0) for ach in resource_achievements)
        hard_resource = sum(hard_stats["achievement_counts"].get(ach, 0) for ach in resource_achievements)
        
        if rr_resource > hard_resource:
            print(f"✅ Resource-dependent achievements: {rr_resource} vs {hard_resource} (+{rr_resource - hard_resource})")
        else:
            print(f"❌ Resource-dependent achievements: {rr_resource} vs {hard_resource}")


if __name__ == "__main__":
    asyncio.run(main())