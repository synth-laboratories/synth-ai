#!/usr/bin/env python3
"""
Generate Fine-tuning Data for Gemini Models
===========================================
This script generates high-quality trajectories from Crafter using Gemini models
and converts them to JSONL format suitable for Vertex AI fine-tuning.
"""

import asyncio
import json
import uuid
import argparse
import toml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
import sys
import os
import numpy as np
from collections import defaultdict
import time
from tqdm.asyncio import tqdm_asyncio
from httpx import AsyncClient

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from synth_ai.lm.core.main import LM
from synth_ai.lm.tools.base import BaseTool
from pydantic import BaseModel, Field

# Import TaskInstance and related classes
from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    Task,
    TaskInstance,
)
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata

# Import trace evaluation utilities
sys.path.append(str(Path(__file__).parent))
from trace_eval import evaluate_trace, WEIGHTS
from filter_traces_sft import load_trace, extract_trajectory_score, extract_llm_calls


# --- Helper Functions ---
def parse_observation_text(obs_text: str) -> Dict[str, Any]:
    """Parse structured observation from text format."""
    obs_data = {
        "health": 10,
        "hunger": 10,
        "thirst": 10,
        "inventory": {},
        "achievements_dict": {},
        "player_position": [0, 0],
        "semantic_map": [],
        "done": False
    }
    
    if not obs_text:
        return obs_data
    
    lines = obs_text.strip().split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse stats
        if "Health:" in line:
            try:
                health = line.split(":")[1].strip().split("/")[0]
                obs_data["health"] = int(health)
            except:
                pass
        elif "Hunger:" in line:
            try:
                hunger = line.split(":")[1].strip().split("/")[0]
                obs_data["hunger"] = int(hunger)
            except:
                pass
        elif "Thirst:" in line:
            try:
                thirst = line.split(":")[1].strip().split("/")[0]
                obs_data["thirst"] = int(thirst)
            except:
                pass
        elif "Position:" in line:
            try:
                pos_str = line.split(":")[1].strip()
                x, y = pos_str.strip("()").split(",")
                obs_data["player_position"] = [int(x), int(y)]
            except:
                pass
        elif "Inventory:" in line:
            current_section = "inventory"
        elif "Achievements:" in line:
            current_section = "achievements"
        elif current_section == "inventory" and " - " in line:
            try:
                item, count = line.split(" - ")
                item = item.strip().strip("-").strip()
                count = int(count.split(":")[1].strip())
                obs_data["inventory"][item] = count
            except:
                pass
        elif current_section == "achievements" and line:
            # Parse achievements list
            achievements = line.split(", ")
            for ach in achievements:
                ach = ach.strip()
                if ach:
                    obs_data["achievements_dict"][ach] = True
    
    return obs_data


# --- Configuration ---
class GenerationConfig:
    """Configuration for fine-tuning data generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.model_name = "gemini-2.5-flash"  # Best Gemini model for reasoning
        self.num_rollouts = 100
        self.max_turns = 30
        self.difficulty = "easy"
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.traces_dir = Path("traces_gemini")
        self.ft_data_dir = Path("ft_data_gemini")
        
        # Quality filtering
        self.min_score_threshold = 2.0  # Minimum trajectory score
        self.min_achievements = 3  # Minimum achievements required
        self.enable_thinking = True  # Enable thinking/reasoning
        self.thinking_budget = 15000  # Token budget for thinking
        
        # Load from TOML if provided
        if config_path and os.path.exists(config_path):
            self.load_from_toml(config_path)
            
    def load_from_toml(self, config_path: str):
        """Load configuration from TOML file."""
        config = toml.load(config_path)
        
        # Extract generation settings
        gen_config = config.get("generation", {})
        self.model_name = gen_config.get("model_name", self.model_name)
        self.num_rollouts = gen_config.get("num_rollouts", self.num_rollouts)
        self.max_turns = gen_config.get("max_turns", self.max_turns)
        self.difficulty = gen_config.get("difficulty", self.difficulty)
        self.seed = gen_config.get("seed", self.seed)
        
        # Extract service settings
        service_config = config.get("service", {})
        self.service_base_url = service_config.get("base_url", self.service_base_url)
        self.service_timeout = service_config.get("timeout", self.service_timeout)
        
        # Extract quality settings
        quality_config = config.get("quality", {})
        self.min_score_threshold = quality_config.get("min_score_threshold", self.min_score_threshold)
        self.min_achievements = quality_config.get("min_achievements", self.min_achievements)
        self.enable_thinking = quality_config.get("enable_thinking", self.enable_thinking)
        self.thinking_budget = quality_config.get("thinking_budget", self.thinking_budget)


# --- Crafter Action Tool ---
class CrafterAction(BaseTool):
    """Tool for performing actions in Crafter environment."""
    
    name: str = "crafter_action"
    description: str = "Perform an action in the Crafter environment"
    params: List[tuple] = [
        ("action", "str", "The action to perform (e.g., 'move_north', 'collect', 'craft_wood_pickaxe')")
    ]
    
    def __init__(self, instance_id: str, client: AsyncClient):
        super().__init__()
        self.instance_id = instance_id
        self.client = client
        self.base_url = "http://localhost:8901"
        
        # Action mapping from string to integer
        self.action_map = {
            "noop": 0,
            "move_north": 1,
            "move_south": 2,
            "move_east": 3,
            "move_west": 4,
            "attack": 5,
            "collect": 6,
            "craft_wood_pickaxe": 7,
            "craft_stone_pickaxe": 8,
            "craft_iron_pickaxe": 9,
            "craft_wood_sword": 10,
            "craft_stone_sword": 11,
            "craft_iron_sword": 12,
            "eat": 13,
            "drink": 14,
            "sleep": 15,
            "place_stone": 16,
            "place_table": 17,
            "place_furnace": 18,
            "place_plant": 19,
        }
    
    def _action_to_int(self, action: str) -> int:
        """Convert action string to integer."""
        return self.action_map.get(action.lower(), 0)
    
    async def _run(self, action: str) -> Dict[str, Any]:
        """Execute action in environment."""
        response = await self.client.post(
            f"{self.base_url}/env/CrafterClassic/step",
            json={"env_id": self.instance_id, "tool_calls": [{
                "tool_name": "interact",
                "tool_call_id": str(uuid.uuid4()),
                "tool_args": {"action": self._action_to_int(action)}
            }]}
        )
        response.raise_for_status()
        result = response.json()
        
        # Return the full result for the agent to process
        return result


# --- Gemini Agent ---
class GeminiCrafterAgent:
    """Agent that plays Crafter using Gemini models via synth-ai LM."""
    
    def __init__(self, model_name: str, instance_id: str, client: AsyncClient):
        self.model_name = model_name
        self.instance_id = instance_id
        self.client = client
        
        # Initialize LM with Gemini model
        self.lm = LM(
            model_name=model_name,
            formatting_model_name=model_name,
            temperature=0.7  # Use some temperature for diversity
        )
        
        # Create action tool
        self.action_tool = CrafterAction(instance_id, client)
        
        # Initialize conversation history
        self.messages = []
        
        # System prompt
        self.system_prompt = """You are an expert Crafter player. Your goal is to achieve as many objectives as possible in the game.

Key objectives (achievements) in order of importance:
1. Basic survival: collect resources, eat when hungry, drink when thirsty
2. Tool progression: craft pickaxe → stone pickaxe → iron pickaxe
3. Advanced goals: make iron sword, defeat enemies

Action format: Use the crafter_action tool with one of these actions:
- Movement: move_north, move_south, move_east, move_west
- Resource gathering: collect (gathers wood/stone/etc), attack (mines harder materials)
- Crafting: craft_wood_pickaxe, craft_stone_pickaxe, craft_iron_pickaxe, craft_wood_sword, craft_stone_sword, craft_iron_sword
- Survival: eat, drink, sleep
- Placing: place_stone, place_table, place_furnace, place_plant

Tips:
- Start by collecting wood (stand near trees and use 'collect')
- Craft a wood pickaxe early to mine stone
- Monitor your health, hunger, and thirst
- Explore to find water, coal, and iron
- Use the semantic map to navigate efficiently"""
        
        # Add system message
        self.messages.append({"role": "system", "content": self.system_prompt})
    
    def _format_observation(self, obs_data: Dict[str, Any]) -> str:
        """Format observation data into readable text."""
        lines = ["=== Current State ==="]
        
        # Stats
        lines.append(f"Position: ({obs_data.get('player_position', [0, 0])[0]}, {obs_data.get('player_position', [0, 0])[1]})")
        lines.append(f"Health: {obs_data.get('health', 0)}/10")
        lines.append(f"Hunger: {obs_data.get('hunger', 0)}/10")
        lines.append(f"Thirst: {obs_data.get('thirst', 0)}/10")
        
        # Inventory
        inventory = obs_data.get('inventory', {})
        if inventory:
            lines.append("\nInventory:")
            for item, count in inventory.items():
                if count > 0:
                    lines.append(f"  - {item}: {count}")
        
        # Achievements
        achievements = obs_data.get('achievements_dict', {})
        unlocked = [k for k, v in achievements.items() if v]
        if unlocked:
            lines.append(f"\nAchievements: {', '.join(unlocked)}")
        
        # Local view (simplified)
        lines.append("\nNearby (5x5 grid around you):")
        semantic_map = obs_data.get('semantic_map', [])
        if semantic_map:
            # Get center region of semantic map
            # Assuming semantic map is flattened, reconstruct as 2D
            map_size = int(np.sqrt(len(semantic_map)))
            if map_size * map_size == len(semantic_map):
                map_2d = np.array(semantic_map).reshape(map_size, map_size)
                center = map_size // 2
                view_radius = 2
                
                # Simple ID to symbol mapping
                id_to_symbol = {
                    0: '.',   # void/empty
                    1: 'G',   # grass
                    2: 'T',   # tree
                    3: 'S',   # stone
                    4: 'W',   # water
                    5: 'C',   # coal
                    6: 'I',   # iron
                    7: '@',   # player
                    8: 'E',   # enemy
                    9: 'F',   # furnace
                    10: 'P',  # plant
                }
                
                for dy in range(-view_radius, view_radius + 1):
                    row = []
                    for dx in range(-view_radius, view_radius + 1):
                        y, x = center + dy, center + dx
                        if 0 <= y < map_size and 0 <= x < map_size:
                            cell_id = int(map_2d[y, x])
                            symbol = id_to_symbol.get(cell_id, '?')
                            if dy == 0 and dx == 0:
                                symbol = '@'  # Player position
                            row.append(symbol)
                        else:
                            row.append(' ')
                    lines.append('  ' + ' '.join(row))
        
        return '\n'.join(lines)
    
    async def step(self, obs_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Take a step in the environment."""
        # Format observation
        obs_text = self._format_observation(obs_data)
        
        # Add observation to conversation
        self.messages.append({"role": "user", "content": obs_text})
        
        # Get action from LM with tool
        response = await self.lm.ainvoke(
            self.messages,
            tools=[self.action_tool],
            tool_choice="required"
        )
        
        # Extract action from response
        action = None
        thinking = None
        
        # Handle response based on type
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Tool was called
            tool_call = response.tool_calls[0]
            action = tool_call.function.arguments.get('action', 'noop')
            
            # Add assistant message
            self.messages.append({
                "role": "assistant",
                "content": response.content or f"Taking action: {action}",
                "tool_calls": response.tool_calls
            })
        else:
            # No tool call, extract action from text
            content = response.content if hasattr(response, 'content') else str(response)
            self.messages.append({"role": "assistant", "content": content})
            action = "noop"
        
        # Extract thinking if available
        if hasattr(response, '_raw_response'):
            raw = response._raw_response
            if isinstance(raw, dict) and 'thinking' in raw:
                thinking = raw['thinking']
        
        # Execute action
        result = await self.action_tool._run(action)
        
        # Add tool response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            self.messages.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": response.tool_calls[0].id
            })
        
        return action, {"thinking": thinking} if thinking else {}


# --- Main Generation Functions ---
async def generate_trajectory(config: GenerationConfig, instance_num: int) -> Optional[Dict[str, Any]]:
    """Generate a single trajectory using Gemini model."""
    async with AsyncClient(timeout=config.service_timeout) as client:
        try:
            # Create task instance
            task_instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions="Survive and unlock achievements. Focus on collecting resources, crafting tools, and progressing through the game."
                ),
                intent=Intent(
                    rubric={"goal": "Unlock as many achievements as possible."},
                    gold_trajectories=None,
                    gold_state_diff={},
                    deterministic_eval_functions=[]
                ),
                metadata=CrafterTaskInstanceMetadata(
                    difficulty=config.difficulty,
                    seed=config.seed + instance_num,
                    num_trees_radius=4,
                    num_cows_radius=2,
                    num_hostiles_radius=0 if config.difficulty == "easy" else 2,
                    world_config="normal"
                ),
                is_reproducible=True,
                initial_engine_snapshot=None  # will be filled lazily when env starts
            )
            
            # Initialize environment
            create_response = await client.post(
                f"{config.service_base_url}/env/CrafterClassic/initialize",
                json={"task_instance": await task_instance.serialize()}
            )
            create_response.raise_for_status()
            env_data = create_response.json()
            instance_id = env_data["env_id"]
            
            print(f"🎮 Instance {instance_num}: Created {instance_id}")
            
            # Get initial observation
            obs_data = env_data.get("observations", [{}])[0]
            
            # Parse the observation to get structured data
            if "human_observation" in obs_data:
                obs_text = obs_data["human_observation"]
                # Parse structured data from observation text
                obs_data = parse_observation_text(obs_text)
                obs_data["raw_text"] = obs_text
            
            # Create agent
            agent = GeminiCrafterAgent(
                model_name=config.model_name,
                instance_id=instance_id,
                client=client
            )
            
            # Track trajectory data
            trajectory = {
                "instance_id": instance_id,
                "instance_num": instance_num,
                "model": config.model_name,
                "start_time": datetime.now().isoformat(),
                "actions": [],
                "observations": [],
                "llm_calls": [],
                "achievements": {},
                "final_score": 0.0
            }
            
            # Run episode
            for turn in range(config.max_turns):
                # Get action from agent
                action, metadata = await agent.step(obs_data)
                
                # Store LLM call data
                llm_call = {
                    "turn": turn,
                    "messages": agent.messages[-3:],  # Last 3 messages (user, assistant, tool)
                    "action": action,
                    "metadata": metadata
                }
                trajectory["llm_calls"].append(llm_call)
                
                # Store action and observation
                trajectory["actions"].append(action)
                trajectory["observations"].append(obs_data)
                
                # Check if done
                if obs_data.get("done", False):
                    print(f"✅ Instance {instance_num}: Episode done at turn {turn}")
                    break
                
                # Step in environment and get next observation
                step_response = await client.post(
                    f"{config.service_base_url}/env/CrafterClassic/step",
                    json={"env_id": instance_id, "tool_calls": [{
                        "tool_name": "interact",
                        "tool_call_id": str(uuid.uuid4()),
                        "tool_args": {"action": agent.action_tool._action_to_int(action)}
                    }]}
                )
                step_response.raise_for_status()
                step_data = step_response.json()
                
                # Extract observation
                if "observations" in step_data and step_data["observations"]:
                    obs = step_data["observations"][0]
                    if "human_observation" in obs:
                        obs_data = parse_observation_text(obs["human_observation"])
                        obs_data["done"] = step_data.get("done", False)
                    else:
                        obs_data = {"done": step_data.get("done", False)}
                else:
                    obs_data = {"done": True}
            
            # Get final achievements
            trajectory["achievements"] = obs_data.get("achievements_dict", {})
            trajectory["end_time"] = datetime.now().isoformat()
            
            # Also get achievements from last observation if available
            if trajectory["observations"] and "achievements_dict" in trajectory["observations"][-1]:
                trajectory["achievements"].update(trajectory["observations"][-1]["achievements_dict"])
            
            # Calculate score
            unlocked_achievements = sum(1 for v in trajectory["achievements"].values() if v)
            trajectory["final_score"] = float(unlocked_achievements)
            
            print(f"📊 Instance {instance_num}: Score={trajectory['final_score']:.1f}, Achievements={unlocked_achievements}")
            
            # Terminate instance
            await client.post(
                f"{config.service_base_url}/env/CrafterClassic/terminate",
                json={"env_id": instance_id}
            )
            
            return trajectory
            
        except Exception as e:
            print(f"❌ Instance {instance_num}: Error - {e}")
            return None


async def generate_all_trajectories(config: GenerationConfig) -> List[Dict[str, Any]]:
    """Generate multiple trajectories concurrently."""
    print(f"\n🚀 Generating {config.num_rollouts} trajectories with {config.model_name}")
    
    # Create tasks
    tasks = [generate_trajectory(config, i) for i in range(config.num_rollouts)]
    
    # Run with progress bar
    trajectories = []
    with tqdm_asyncio(total=config.num_rollouts, desc="Generating") as pbar:
        for coro in asyncio.as_completed(tasks):
            trajectory = await coro
            if trajectory:
                trajectories.append(trajectory)
            pbar.update(1)
    
    return trajectories


def filter_high_quality_trajectories(trajectories: List[Dict[str, Any]], 
                                   min_score: float = 2.0,
                                   min_achievements: int = 3) -> List[Dict[str, Any]]:
    """Filter trajectories based on quality criteria."""
    filtered = []
    
    for traj in trajectories:
        # Count achievements
        achievements = traj.get("achievements", {})
        num_achievements = sum(1 for v in achievements.values() if v)
        
        # Calculate score (could be more sophisticated)
        score = traj.get("final_score", 0.0)
        
        # Apply filters
        if score >= min_score and num_achievements >= min_achievements:
            filtered.append(traj)
    
    print(f"\n📊 Filtering Results:")
    print(f"   Total trajectories: {len(trajectories)}")
    if trajectories:
        print(f"   High quality: {len(filtered)} ({len(filtered)/len(trajectories)*100:.1f}%)")
    else:
        print(f"   High quality: 0 (no trajectories generated)")
    
    return filtered


def convert_to_vertex_ai_format(trajectories: List[Dict[str, Any]], output_path: Path):
    """Convert trajectories to Vertex AI fine-tuning format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    
    for traj in trajectories:
        # Extract LLM calls
        for llm_call in traj.get("llm_calls", []):
            messages = llm_call.get("messages", [])
            
            # Skip if not enough messages
            if len(messages) < 2:
                continue
            
            # Convert to Vertex AI format
            # Need user message and assistant response
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]
            
            if user_msg and assistant_msg:
                example = {
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                }
                examples.append(example)
    
    # Write JSONL
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✅ Wrote {len(examples)} examples to {output_path}")
    return len(examples)


def save_trajectories(trajectories: List[Dict[str, Any]], traces_dir: Path):
    """Save trajectories to disk."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    for i, traj in enumerate(trajectories):
        filename = f"trajectory_{i:04d}.json"
        with open(traces_dir / filename, 'w') as f:
            json.dump(traj, f, indent=2)
    
    print(f"💾 Saved {len(trajectories)} trajectories to {traces_dir}")


# --- Main ---
async def main():
    parser = argparse.ArgumentParser(description="Generate Gemini fine-tuning data for Crafter")
    parser.add_argument("--config", type=str, help="Path to TOML config file")
    parser.add_argument("--num-rollouts", type=int, help="Number of rollouts to generate")
    parser.add_argument("--model", type=str, help="Gemini model name")
    parser.add_argument("--filter-only", action="store_true", help="Only filter existing traces")
    parser.add_argument("--min-achievements", type=int, help="Minimum achievements for filtering")
    
    args = parser.parse_args()
    
    # Load config
    config = GenerationConfig(args.config)
    
    # Override with command line args
    if args.num_rollouts:
        config.num_rollouts = args.num_rollouts
    if args.model:
        config.model_name = args.model
    if args.min_achievements:
        config.min_achievements = args.min_achievements
    
    if args.filter_only:
        # Filter existing trajectories
        print("🔍 Filtering existing trajectories...")
        
        # Load trajectories
        trajectories = []
        for trace_file in sorted(config.traces_dir.glob("*.json")):
            with open(trace_file) as f:
                trajectories.append(json.load(f))
        
        # Filter
        filtered = filter_high_quality_trajectories(
            trajectories,
            min_score=config.min_score_threshold,
            min_achievements=config.min_achievements
        )
        
        # Convert to JSONL
        output_path = config.ft_data_dir / "crafter_gemini_ft.jsonl"
        num_examples = convert_to_vertex_ai_format(filtered, output_path)
        
        print(f"\n🎯 Summary:")
        print(f"   Filtered trajectories: {len(filtered)}")
        print(f"   Total training examples: {num_examples}")
        
    else:
        # Generate new trajectories
        trajectories = await generate_all_trajectories(config)
        
        # Save all trajectories
        save_trajectories(trajectories, config.traces_dir)
        
        # Filter high quality
        filtered = filter_high_quality_trajectories(
            trajectories,
            min_score=config.min_score_threshold,
            min_achievements=config.min_achievements
        )
        
        # Convert to JSONL
        output_path = config.ft_data_dir / "crafter_gemini_ft.jsonl"
        num_examples = convert_to_vertex_ai_format(filtered, output_path)
        
        print(f"\n🎯 Summary:")
        print(f"   Generated trajectories: {len(trajectories)}")
        print(f"   High quality trajectories: {len(filtered)}")
        print(f"   Total training examples: {num_examples}")
        print(f"   Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())