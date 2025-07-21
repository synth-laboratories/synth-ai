#!/usr/bin/env python3
"""
Test script to run ReAct agents against Crafter environment on synth service (port 8901)
Tests on multiple easy Crafter instances with enhanced debugging
"""

import asyncio
import json
import uuid
import math
import argparse
import toml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel, Field
from httpx import AsyncClient
import sys
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool
import numpy as np


# --- Configuration Class ---
class CrafterConfig:
    """Configuration for Crafter evaluation."""

    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.model_name: Optional[str] = None  # Must be provided via config or CLI
        self.num_instances = 3
        self.max_turns = 20
        self.difficulty = "easy"
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.save_traces = True
        self.save_detailed_results = True
        self.verbose = False

        # Custom OpenAI endpoint support
        self.custom_openai_base_url = None  # e.g., "https://lora-inference-service-xyz.modal.run"
        self.custom_openai_api_key = "dummy"  # Default dummy key for custom endpoints

        # Load from TOML if provided
        if config_path and os.path.exists(config_path):
            self.load_from_toml(config_path)

        # Fail fast if no model name provided
        # Configure custom OpenAI endpoint if specified
        self._configure_custom_openai()

    def load_from_toml(self, config_path: str):
        """Load configuration from TOML file."""
        config = toml.load(config_path)

        # Extract eval settings
        eval_config = config.get("eval", {})
        self.model_name = eval_config.get("model_name", self.model_name)
        self.num_instances = eval_config.get("episodes", self.num_instances)
        self.max_turns = eval_config.get("max_steps", self.max_turns)
        self.difficulty = eval_config.get("difficulty", self.difficulty)
        self.seed = eval_config.get("seed", self.seed)

        # Extract service settings
        service_config = config.get("service", {})
        self.service_base_url = service_config.get("base_url", self.service_base_url)
        self.service_timeout = service_config.get("timeout", self.service_timeout)

        # Extract output settings
        output_config = config.get("output", {})
        self.save_traces = output_config.get("save_traces", self.save_traces)
        self.save_detailed_results = output_config.get(
            "save_detailed_results", self.save_detailed_results
        )

        # Extract custom OpenAI endpoint settings
        openai_config = config.get("openai", {})
        self.custom_openai_base_url = openai_config.get("base_url", self.custom_openai_base_url)
        self.custom_openai_api_key = openai_config.get("api_key", self.custom_openai_api_key)

    def _configure_custom_openai(self):
        """Configure environment variables for custom OpenAI endpoint if specified."""
        if self.custom_openai_base_url:
            # Ensure the base URL ends with /v1 for OpenAI compatibility
            base_url = self.custom_openai_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

            # Set environment variables for OpenAI SDK
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_KEY"] = self.custom_openai_api_key

            print(f"🔧 Configured custom OpenAI endpoint: {base_url}")
            print(f"   API Key: {self.custom_openai_api_key}")

            # Auto-detect if this looks like a fine-tuned model and add ft: regex support
            if self.model_name and (
                self.model_name.startswith("ft:") or "lora" in self.model_name.lower()
            ):
                self._add_ft_regex_support()

    def _add_ft_regex_support(self):
        """Add ft: regex pattern to OpenAI naming regexes if not already present."""
        try:
            import re
            from synth_ai.zyk.lms.core import vendor_clients

            # Check if ft: pattern already exists
            ft_pattern = re.compile(r"^ft:.*$")
            if not any(
                pattern.pattern == ft_pattern.pattern
                for pattern in vendor_clients.openai_naming_regexes
            ):
                # Add ft: pattern at the beginning to catch all fine-tuned models
                vendor_clients.openai_naming_regexes.insert(0, ft_pattern)
                print(f"✅ Added ft:* regex pattern for fine-tuned model support")
        except Exception as e:
            print(f"⚠️  Warning: Could not add ft: regex pattern: {e}")

    def set_custom_endpoint(self, base_url: str, api_key: str = "dummy"):
        """Programmatically set custom OpenAI endpoint."""
        self.custom_openai_base_url = base_url
        self.custom_openai_api_key = api_key
        self._configure_custom_openai()


# --- Global Config ---
config = CrafterConfig()


# --- Helper to build crafter semantic mapping ---
def get_crafter_semantic_mapping():
    """Build the crafter semantic ID to item name mapping."""
    try:
        import crafter
        import itertools

        # Create a dummy env to get ID mappings
        dummyenv = None
        try:
            dummyenv = crafter.Env()
            max_id = (
                max(
                    max(dummyenv._world._mat_ids.values()),
                    max(dummyenv._sem_view._obj_ids.values()),
                )
                + 1
            )
            id_to_item = ["void"] * max_id
            for name, ind in itertools.chain(
                dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()
            ):
                if name is None:
                    clean = "none"
                elif hasattr(name, "__name__"):
                    clean = name.__name__
                else:
                    clean = str(name)
                id_to_item[ind] = clean.lower()
            player_idx = id_to_item.index("player")
            return id_to_item, player_idx
        finally:
            if dummyenv:
                try:
                    dummyenv.close()
                except Exception:
                    pass
            del dummyenv
    except ImportError:
        # Fallback if crafter is not available
        return None, None


def format_semantic_map_view(obs_data: Dict[str, Any], view_size: int = 7) -> str:
    """Format a semantic map view around the player (ASCII)."""
    try:
        # Get mapping list
        id_to_item, _ = get_crafter_semantic_mapping()
        if id_to_item is None:
            return "Map view unavailable (crafter not installed)"

        semantic_map = obs_data.get("semantic_map")
        player_position = obs_data.get("player_position", [0, 0])

        if semantic_map is None:
            return "Map view unavailable (no semantic map data)"

        # Ensure numpy array with 2 dimensions
        sem_arr = np.asarray(semantic_map)
        if sem_arr.ndim == 1:
            # Probably flattened; try to infer square size
            size = int(np.sqrt(sem_arr.size))
            sem_arr = sem_arr.reshape(size, size)
        elif sem_arr.ndim != 2:
            return "Map view unavailable (invalid map dimensionality)"

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
                if token not in {"void", "player"}:
                    visible.add(token)
            rows.append(" ".join(row_tokens))

        map_view = f"\nLocal Map View ({view_size}x{view_size}):\n" + "\n".join(rows)
        if visible:
            map_view += "\nVisible items: " + ", ".join(sorted(visible))
        else:
            map_view += "\nNo special items visible (mostly grass/empty)"
        return map_view
    except Exception as e:
        return f"Map view error: {e}"


# --- Shaped Reward Configuration ---
# K-values for shaped reward calculation: reward = sum(K * log(count)) for each achievement
ACHIEVEMENT_K_VALUES = {
    "collect_coal": 3.0,
    "collect_diamond": 100.0,
    "collect_drink": 0.1,
    "collect_iron": 10.0,
    "collect_sapling": 0.1,
    "collect_stone": 1.0,
    "collect_wood": 1.0,
    "defeat_skeleton": 1.0,
    "defeat_zombie": 1.0,
    "eat_cow": 1.0,
    "eat_plant": 0.1,
    "make_iron_pickaxe": 30.0,
    "make_iron_sword": 30.0,
    "make_stone_pickaxe": 10.0,
    "make_stone_sword": 10.0,
    "make_wood_pickaxe": 3.0,
    "make_wood_sword": 3.0,
    "place_furnace": 10.0,
    "place_plant": 0.1,
    "place_stone": 1.0,
    "place_table": 3.0,
    "wake_up": 0.1,
}


# --- Tool Definitions ---
class CrafterActionArgs(BaseModel):
    """Arguments for crafter actions."""

    actions: List[str] = Field(
        description="List of 1-5 action names to execute in sequence (e.g., ['move_up', 'do', 'mine_down'])"
    )
    reasoning: str = Field(description="Brief explanation of why these actions were chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


class CrafterActionTool(BaseTool):
    """Tool for performing actions in the Crafter environment."""

    name: str = "interact"
    arguments: type[BaseModel] = CrafterActionArgs
    description: str = "Perform 1-5 actions in sequence in the Crafter environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Shaped Reward Helper ---
def calculate_shaped_reward(achievement_counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculate shaped reward using K * log(count) for each achievement."""
    total_reward = 0.0
    reward_breakdown = {}

    for achievement, count in achievement_counts.items():
        if count > 0 and achievement in ACHIEVEMENT_K_VALUES:
            k_value = ACHIEVEMENT_K_VALUES[achievement]
            # Use log(count + 1) to handle count=0 case gracefully
            reward_contribution = k_value * math.log(count + 1)
            total_reward += reward_contribution
            reward_breakdown[achievement] = {
                "count": count,
                "k_value": k_value,
                "contribution": reward_contribution,
            }

    return {"total_shaped_reward": total_reward, "breakdown": reward_breakdown}


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""

    def __init__(self, llm: LM, max_turns: int = 20, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"

        # Define tools in OpenAI format
        self.tools = [
            CrafterActionTool(),
            TerminateTool(),
        ]

    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision based on observation."""
        # Create conversation context
        context = f"Turn {turn + 1}/{self.max_turns}\n\n{obs}"
        # Generate response using LLM
        response_obj = await self.llm.respond_async(
            system_message=system_message, user_message=context, tools=self.tools
        )

        tool_calls = response_obj.tool_calls

        # Handle case where tool_calls is None or empty (graceful fallback)
        if not tool_calls:
            if self.verbose:
                print(f"[WARNING] No tool calls returned by LLM, using default action")
            return {
                "name": "interact",
                "parameters": {
                    "actions": ["do"],
                    "reasoning": "Default action - no tool call received",
                },
            }

        tool_call_data = tool_calls[0]

        # Handle both dict and object formats
        if isinstance(tool_call_data, dict):
            tool_name = tool_call_data["function"]["name"]
            tool_args_str = tool_call_data["function"]["arguments"]
        else:
            tool_name = tool_call_data.function.name
            tool_args_str = tool_call_data.function.arguments

        tool_arguments = json.loads(tool_args_str)

        return {"name": tool_name, "parameters": tool_arguments}


# --- Crafter ReAct Agent ---
class CrafterReActAgent(BaseReActAgent):
    """ReAct agent for Crafter environment."""

    def __init__(self, llm: LM, max_turns: int = 20, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "crafter-react-agent"

    def get_system_message(self) -> str:
        return """You are CrafterAgent playing Crafter survival environment. Your goal is to unlock as many achievements as possible while staying alive.

You will see a semantic map view showing your surroundings. Use this to navigate toward resources.

Key mechanics:
• 'do' action: collect wood from trees, stone from deposits, food from cows/plants
• 'do' does nothing on grass/water - move to find resources first
• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools
• Sleep when energy low to restore and unlock wake_up achievement
• Use semantic map view to navigate toward resources you can see

Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop

Strategy:
1. Look at the semantic map to see what's around you
2. Move toward trees to collect wood with 'do'
3. Once you have wood, place a table to enable crafting
4. Make a wood pickaxe to collect stone more efficiently
5. Progress to stone pickaxe, then iron tools
6. Eat food when health is low, sleep when energy is low

You should provide 1-5 actions in sequence for efficient gameplay. Use the semantic map view to navigate toward visible resources.

Example good action sequences:
- ['move_right', 'move_right', 'do'] (move to tree and collect wood)
- ['place_table', 'make_wood_pickaxe'] (craft progression)
- ['move_up', 'do', 'move_down', 'do'] (collect from multiple resources)

Be strategic and use the map view to find resources! Focus on unlocking achievements."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for Crafter with rich context."""
        # Extract key information from observation
        health = obs.get("health", 0)
        inventory = obs.get("inventory", {})

        # Extract health from inventory if not in main obs
        if health == 0 and "health" in inventory:
            health = inventory["health"]

        # Format inventory items (exclude health since we show it separately)
        inventory_items = []
        for item, count in inventory.items():
            if count > 0 and item != "health":
                inventory_items.append(f"{item}: {count}")

        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"

        # Get achievements
        achievements = obs.get("achievements") or obs.get("achievements_status", {})
        unlocked_achievements = [name for name, unlocked in achievements.items() if unlocked]
        achievements_str = ", ".join(unlocked_achievements) if unlocked_achievements else "none"

        # Get position and other state
        position = obs.get("position", [0, 0])
        player_position = obs.get("player_position", position)
        player_direction = obs.get("player_direction", [0, 1])
        num_steps = obs.get("num_steps_taken", 0)

        # Check termination status
        terminated = obs.get("terminated", False)

        # Get semantic map view
        map_view = format_semantic_map_view(obs, view_size=5)

        return (
            f"Crafter Game State:\n"
            f"Step: {num_steps}\n"
            f"Health: {health}\n"
            f"Position: {player_position}\n"
            f"Direction: {player_direction}\n"
            f"Inventory: {inventory_str}\n"
            f"Achievements: {achievements_str}\n"
            f"Terminated: {terminated}\n"
            f"{map_view}\n\n"
            f"Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop\n\n"
            f"Key mechanics:\n"
            f"• 'do' action: collect wood from trees, stone from deposits, food from cows/plants\n"
            f"• 'do' does nothing on grass/water - move to find resources\n"
            f"• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools\n"
            f"• Sleep when energy low to restore and unlock wake_up achievement\n\n"
            f"Choose 1-5 actions to execute in sequence. Focus on exploring to find resources and crafting tools to unlock achievements."
        )


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient, agent: CrafterReActAgent, task_instance, instance_num: int
) -> Dict[str, Any]:
    """Run a single Crafter episode and return episode metrics."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/CrafterClassic/initialize",
            json={"task_instance": await task_instance.serialize()},
        )

        if create_resp.status_code != 200:
            print(
                f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}"
            )
            return {
                "eval_metric": 0.0,
                "rubric": {},
                "total_reward": 0.0,
                "num_achievements": 0,
                "terminated": False,
                "error": True,
            }

        env_id = create_resp.json()["env_id"]

        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)

        # DEBUG: Print initial state (minimal)
        print(
            f"\n  Instance {instance_num}: Starting Crafter survival ({task_instance.metadata.difficulty}, {agent.max_turns} turns max)"
        )

        # Track episode metrics
        total_reward = 0.0
        final_achievements = {}
        num_achievements = 0
        terminated = False
        rollout_length = 0

        # Run episode
        for turn in range(agent.max_turns):
            try:
                # Get agent decision
                action = await agent.decide(formatted_obs, agent.get_system_message(), turn)
                # print(f"  ✅ Agent decision received: {action}")

                # # DEBUG: Print agent decision with safer access
                # try:
                #     actions = action.get('parameters', {}).get('actions', action.get('arguments', {}).get('actions', []))
                #     reasoning = action.get('parameters', {}).get('reasoning', action.get('arguments', {}).get('reasoning', 'no reasoning'))
                #     #print(f"  Turn {turn+1}: Agent chose {actions} - {reasoning}")
                # except Exception as e:
                #     print(f"  Turn {turn+1}: Agent action structure: {action}")
                #     print(f"  Error parsing action: {e}")

                # Check for termination
                if action["name"] == "terminate":
                    reason = action.get("parameters", {}).get(
                        "reason", action.get("arguments", {}).get("reason", "no reason given")
                    )
                    print(f"  Agent terminated: {reason}")
                    break

                # Execute actions in environment with safer access
                action_sequence = action.get("parameters", {}).get(
                    "actions", action.get("arguments", {}).get("actions", [])
                )
                if not action_sequence:
                    print(f"  ⚠️  No actions found in: {action}")
                    continue

                # Convert action names to integers using the proper action map
                # Define the proper Crafter action mapping
                CRAFTER_ACTION_MAP = {
                    "noop": 0,
                    "move_left": 1,
                    "move_right": 2,
                    "move_up": 3,
                    "move_down": 4,
                    "do": 5,
                    "sleep": 6,
                    "place_stone": 7,
                    "place_table": 8,
                    "place_furnace": 9,
                    "place_plant": 10,
                    "make_wood_pickaxe": 11,
                    "make_stone_pickaxe": 12,
                    "make_iron_pickaxe": 13,
                    "make_wood_sword": 14,
                    "make_stone_sword": 15,
                    "make_iron_sword": 16,
                }

                action_ints = []
                for action_name in action_sequence:
                    if action_name in CRAFTER_ACTION_MAP:
                        action_int = CRAFTER_ACTION_MAP[action_name]
                    else:
                        action_int = 0  # Default to noop
                    action_ints.append(action_int)

                # Execute each action individually (Crafter expects single actions)
                for i, action_int in enumerate(action_ints):
                    step_resp = await client.post(
                        f"/env/CrafterClassic/step",
                        json={
                            "env_id": env_id,
                            "request_id": str(uuid.uuid4()),
                            "action": {
                                "tool_calls": [{"tool": "interact", "args": {"action": action_int}}]
                            },
                        },
                    )

                    if step_resp.status_code != 200:
                        print(
                            f"    ❌ Action {i + 1} failed: {step_resp.status_code}: {step_resp.text}"
                        )
                        break

                    # Update observation after each action
                    obs = step_resp.json()["observation"]

                # Check final response status
                if step_resp.status_code != 200:
                    break

                # Show final state after all actions
                formatted_obs = agent.format_observation(obs)
                step_count = obs.get("num_steps_taken", 0)
                rollout_length = step_count
                position = obs.get("player_position", [0, 0])
                # print(f"  Turn {turn+1}: Actions completed - Step: {step_count}, Position: {position}")

                # Update history with safer access
                reasoning = action.get("parameters", {}).get(
                    "reasoning", action.get("arguments", {}).get("reasoning", "")
                )
                agent.history.append(f"{', '.join(action_sequence)}: {reasoning[:50]}")

                # Track episode progress - Use the FINAL observation from the last action
                terminated = obs.get("terminated", False)
                step_reward = obs.get("reward", 0.0)
                total_reward += step_reward
                achievements = obs.get("achievements") or obs.get("achievements_status", {})

                # ALWAYS update final_achievements with the latest observation
                final_achievements = achievements

                num_achievements = sum(1 for v in achievements.values() if v) if achievements else 0

                if terminated:
                    print(
                        f"  ✅ Instance {instance_num}: Episode completed! Achievements: {num_achievements}, Total reward: {total_reward:.3f}"
                    )
                    break

            except Exception as e:
                print(f"  ❌ Error in turn {turn + 1}: {e}")
                import traceback

                traceback.print_exc()
                break

        # Cleanup
        await client.post(f"/env/CrafterClassic/terminate", json={"env_id": env_id})

        # Calculate K-weighted achievement reward
        achievement_reward = 0.0
        if final_achievements:
            for achievement, unlocked in final_achievements.items():
                if unlocked and achievement in ACHIEVEMENT_K_VALUES:
                    k_value = ACHIEVEMENT_K_VALUES[achievement]
                    achievement_reward += k_value * math.log(2)  # log(1+1) for single achievement

        # Use achievement reward as the total reward
        total_reward = achievement_reward

        # Calculate eval metric and rubric
        eval_metric = float(num_achievements)  # Simple metric: number of achievements

        # Create rubric with specific achievement checks
        rubric = {}
        if final_achievements:
            rubric = {
                "collect_wood": 1.0 if final_achievements.get("collect_wood", False) else 0.0,
                "collect_stone": 1.0 if final_achievements.get("collect_stone", False) else 0.0,
                "collect_coal": 1.0 if final_achievements.get("collect_coal", False) else 0.0,
                "collect_iron": 1.0 if final_achievements.get("collect_iron", False) else 0.0,
                "collect_diamond": 1.0 if final_achievements.get("collect_diamond", False) else 0.0,
                "place_table": 1.0 if final_achievements.get("place_table", False) else 0.0,
                "place_furnace": 1.0 if final_achievements.get("place_furnace", False) else 0.0,
                "make_wood_pickaxe": 1.0
                if final_achievements.get("make_wood_pickaxe", False)
                else 0.0,
                "make_stone_pickaxe": 1.0
                if final_achievements.get("make_stone_pickaxe", False)
                else 0.0,
                "make_iron_pickaxe": 1.0
                if final_achievements.get("make_iron_pickaxe", False)
                else 0.0,
                "make_wood_sword": 1.0 if final_achievements.get("make_wood_sword", False) else 0.0,
                "make_stone_sword": 1.0
                if final_achievements.get("make_stone_sword", False)
                else 0.0,
                "make_iron_sword": 1.0 if final_achievements.get("make_iron_sword", False) else 0.0,
                "defeat_skeleton": 1.0 if final_achievements.get("defeat_skeleton", False) else 0.0,
                "defeat_zombie": 1.0 if final_achievements.get("defeat_zombie", False) else 0.0,
                "wake_up": 1.0 if final_achievements.get("wake_up", False) else 0.0,
                "eat_cow": 1.0 if final_achievements.get("eat_cow", False) else 0.0,
                "eat_plant": 1.0 if final_achievements.get("eat_plant", False) else 0.0,
            }
        else:
            # Default rubric with all zeros
            rubric = {
                "collect_wood": 0.0,
                "collect_stone": 0.0,
                "collect_coal": 0.0,
                "collect_iron": 0.0,
                "collect_diamond": 0.0,
                "place_table": 0.0,
                "place_furnace": 0.0,
                "make_wood_pickaxe": 0.0,
                "make_stone_pickaxe": 0.0,
                "make_iron_pickaxe": 0.0,
                "make_wood_sword": 0.0,
                "make_stone_sword": 0.0,
                "make_iron_sword": 0.0,
                "defeat_skeleton": 0.0,
                "defeat_zombie": 0.0,
                "wake_up": 0.0,
                "eat_cow": 0.0,
                "eat_plant": 0.0,
            }

        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "total_reward": total_reward,
            "num_achievements": num_achievements,
            "achievements": final_achievements,
            "rollout_length": rollout_length,
            "terminated": terminated,
            "error": False,
        }

    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback

        traceback.print_exc()
        return {
            "eval_metric": 0.0,
            "rubric": {},
            "total_reward": 0.0,
            "num_achievements": 0,
            "terminated": False,
            "error": True,
        }


# --- Batch Evaluation ---
async def evaluate_crafter_batch() -> Dict[str, Any]:
    """Evaluate Crafter agent on multiple easy instances."""
    print(f"🎯 Evaluating Crafter on {config.num_instances} {config.difficulty} instances...")

    llm = LM(model_name=config.model_name, formatting_model_name=config.model_name, temperature=0.0)

    # Get easy task instances using the taskset system
    from synth_ai.environments.examples.crafter_classic.taskset import (
        CrafterTaskInstance,
        CrafterTaskInstanceMetadata,
    )
    from synth_ai.environments.tasks.core import Impetus, Intent

    easy_task_instances = []
    for seed in range(config.num_instances):
        try:
            metadata = CrafterTaskInstanceMetadata(
                difficulty=config.difficulty,
                seed=seed,
                num_trees_radius=5,  # Good for easy difficulty
                num_cows_radius=2,
                num_hostiles_radius=0,  # No hostiles for easy
            )
            task_instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions=f"Survive and unlock achievements in an {config.difficulty} environment."
                ),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            easy_task_instances.append(task_instance)
        except Exception as e:
            print(f"  ⚠️  Failed to create task instance for seed {seed}: {e}")
            continue

    print(f"  📝 Generated {len(easy_task_instances)} {config.difficulty} task instances")

    async with AsyncClient(
        base_url=config.service_base_url, timeout=config.service_timeout
    ) as client:
        # Run trajectories in parallel batches of 4
        batch_size = 4
        all_results = []

        for batch_start in range(0, len(easy_task_instances), batch_size):
            batch_end = min(batch_start + batch_size, len(easy_task_instances))
            batch_instances = easy_task_instances[batch_start:batch_end]

            print(
                f"  🚀 Running batch {batch_start // batch_size + 1} ({len(batch_instances)} episodes)..."
            )

            # Create tasks for this batch
            batch_tasks = []
            for i, task_instance in enumerate(batch_instances):
                agent = CrafterReActAgent(llm, max_turns=config.max_turns, verbose=False)
                batch_tasks.append(
                    run_single_episode(client, agent, task_instance, batch_start + i + 1)
                )

            # Run this batch in parallel with progress bar
            batch_results = await tqdm_asyncio.gather(
                *batch_tasks, desc=f"Batch {batch_start // batch_size + 1}", unit="episode"
            )
            all_results.extend(batch_results)

            print(f"  ✅ Batch {batch_start // batch_size + 1} completed")

        results = all_results

        # Filter out error results
        valid_results = [r for r in results if not r.get("error", False)]

        if not valid_results:
            return {
                "eval_metrics": [],
                "mean_eval_metric": 0.0,
                "mean_rubric": {},
                "num_episodes": 0,
            }

        # Extract eval metrics and rubrics
        eval_metrics = [r["eval_metric"] for r in valid_results]
        mean_eval_metric = sum(eval_metrics) / len(eval_metrics)

        # --- Rollout length statistics ---
        rollout_lengths = [r["rollout_length"] for r in valid_results]
        sorted_lengths = sorted(rollout_lengths)
        n_lengths = len(sorted_lengths)
        # Median (Q2)
        if n_lengths % 2 == 1:
            q2_rollout = sorted_lengths[n_lengths // 2]
        else:
            q2_rollout = (sorted_lengths[n_lengths // 2 - 1] + sorted_lengths[n_lengths // 2]) / 2
        # 90th percentile (P90)
        p90_index = int(0.9 * (n_lengths - 1))
        p90_rollout = sorted_lengths[p90_index]
        max_rollout = sorted_lengths[-1]

        # Calculate mean rubric values
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(r["rubric"].keys())

        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)

        # Calculate shaped reward (training rubric)
        # Count total achievements across all episodes
        achievement_counts = {}
        unique_achievements_per_trajectory = []
        all_unique_achievements = set()

        for result in valid_results:
            achievements = result.get("achievements", {})
            trajectory_achievements = set()
            for achievement, unlocked in achievements.items():
                if unlocked:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
                    trajectory_achievements.add(achievement)
                    all_unique_achievements.add(achievement)
            unique_achievements_per_trajectory.append(trajectory_achievements)

        # Calculate shaped reward using the counts
        shaped_reward_data = calculate_shaped_reward(achievement_counts)

        # Calculate unique achievements by N trajectories
        unique_achievements_by_n = {}
        for n in range(1, len(valid_results) + 1):
            unique_at_n = set()
            for i in range(n):
                unique_at_n.update(unique_achievements_per_trajectory[i])
            unique_achievements_by_n[n] = len(unique_at_n)

        # Create training rubric (normalized shaped reward components)
        training_rubric = {}
        total_episodes = len(valid_results)
        if shaped_reward_data["breakdown"]:
            for achievement, data in shaped_reward_data["breakdown"].items():
                # Normalize by number of episodes for comparison
                training_rubric[achievement] = data["contribution"] / total_episodes

        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "mean_rubric": mean_rubric,
            "achievement_counts": achievement_counts,
            "shaped_reward_data": shaped_reward_data,
            "training_rubric": training_rubric,
            "unique_achievements_per_trajectory": unique_achievements_per_trajectory,
            "all_unique_achievements": all_unique_achievements,
            "unique_achievements_by_n": unique_achievements_by_n,
            "num_episodes": len(valid_results),
            "q2_rollout": q2_rollout,
            "p90_rollout": p90_rollout,
            "max_rollout": max_rollout,
        }


async def main():
    """Run Crafter evaluation."""
    # Configure logging to reduce verbosity
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("google_genai.types").setLevel(logging.WARNING)

    print(f"🎮 Crafter ReAct Agent Evaluation")
    print(f"Model: {config.model_name}")
    print(f"Service: {config.service_base_url}")
    print(f"Instances: {config.num_instances}")
    print(f"Max Turns: {config.max_turns}")
    print(f"Difficulty: {config.difficulty}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Test service health
    async with AsyncClient(base_url=config.service_base_url, timeout=10.0) as client:
        try:
            health_resp = await client.get("/health")
            health_data = health_resp.json()

            if "CrafterClassic" not in health_data.get("supported_environments", []):
                print("❌ CrafterClassic not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return

    # Run evaluation
    try:
        results = await evaluate_crafter_batch()

        print("\n" + "=" * 80)
        print("🏆 FINAL CRAFTER EVALUATION RESULTS")
        print("=" * 80)

        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.1f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")

        # Print standard rubric results
        print(f"\n🎯 STANDARD RUBRIC RESULTS:")
        if results["mean_rubric"]:
            for achievement, score in sorted(results["mean_rubric"].items()):
                print(f"  {achievement}: {score:.2f}")
        else:
            print("  No rubric data available")

        # Print shaped reward results
        print(f"\n🏋️  TRAINING EVAL SCORE (SHAPED REWARD):")
        shaped_data = results.get("shaped_reward_data", {})
        print(f"  Total Shaped Reward: {shaped_data.get('total_shaped_reward', 0.0):.3f}")

        # Print achievement counts and contributions
        achievement_counts = results.get("achievement_counts", {})
        if achievement_counts:
            print(f"\n  Achievement Counts Across All Episodes:")
            for achievement, count in sorted(achievement_counts.items()):
                k_value = ACHIEVEMENT_K_VALUES.get(achievement, 0.0)
                contribution = k_value * math.log(count + 1) if count > 0 else 0.0
                print(
                    f"    {achievement}: {count} times (K={k_value:.1f}, contribution={contribution:.3f})"
                )
        else:
            print("  No achievements unlocked")

        # Print training rubric (normalized contributions)
        print(f"\n🎖️  TRAINING RUBRIC (PER EPISODE):")
        if results.get("training_rubric"):
            for achievement, score in sorted(results["training_rubric"].items()):
                print(f"  {achievement}: {score:.3f}")
        else:
            print("  No training rubric data available")

        # Print unique achievements analysis
        print(f"\n🏆 UNIQUE ACHIEVEMENTS ANALYSIS:")
        all_unique = results.get("all_unique_achievements", set())
        print(f"  Total Unique Achievements Unlocked: {len(all_unique)}")
        if all_unique:
            print(f"  Unique Achievements: {', '.join(sorted(all_unique))}")

        # Print unique achievements by N trajectories
        unique_by_n = results.get("unique_achievements_by_n", {})
        if unique_by_n:
            print(f"\n📊 UNIQUE ACHIEVEMENTS BY N TRAJECTORIES:")
            for n in sorted(unique_by_n.keys()):
                print(f"  After {n} trajectories: {unique_by_n[n]} unique achievements")

        # Calculate average achievements per trajectory
        achievements_per_trajectory = [
            len(achievements)
            for achievements in results.get("unique_achievements_per_trajectory", [])
        ]
        if achievements_per_trajectory:
            avg_achievements = sum(achievements_per_trajectory) / len(achievements_per_trajectory)
            print(f"\n📈 TRAJECTORY ANALYSIS:")
            print(f"  Average Achievements per Trajectory: {avg_achievements:.2f}")
            print(f"  Achievements per Trajectory: {achievements_per_trajectory}")
            print(f"  Best Trajectory: {max(achievements_per_trajectory)} achievements")
            print(f"  Worst Trajectory: {min(achievements_per_trajectory)} achievements")

        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        if results["mean_eval_metric"] >= 3.0:
            print("🎉 Excellent performance - achieving multiple objectives!")
        elif results["mean_eval_metric"] >= 1.0:
            print("✅ Good performance - consistently achieving objectives!")
        elif results["mean_eval_metric"] >= 0.5:
            print("⚠️  Moderate performance - some achievements unlocked")
        else:
            print("📈 Learning phase - focus on basic survival and resource gathering")

        # Output markdown table row for README collation
        print(f"\n📋 MARKDOWN TABLE ROW:")
        print(
            "| Model            | Episodes | Mean Score | Avg Achievements | Unique Achievements | Shaped Reward | Mean K-Score | Q2 Len | P90 Len | Max Len |"
        )
        print(
            "|------------------|----------|------------|------------------|---------------------|---------------|--------------|--------|---------|---------|"
        )
        achievements_per_trajectory = [
            len(achievements)
            for achievements in results.get("unique_achievements_per_trajectory", [])
        ]
        avg_achievements = (
            sum(achievements_per_trajectory) / len(achievements_per_trajectory)
            if achievements_per_trajectory
            else 0.0
        )
        total_unique = len(results.get("all_unique_achievements", set()))
        shaped_reward = results.get("shaped_reward_data", {}).get("total_shaped_reward", 0.0)
        mean_k_score = (
            shaped_reward / results["num_episodes"] if results["num_episodes"] > 0 else 0.0
        )
        q2_rollout = results.get("q2_rollout", 0)
        p90_rollout = results.get("p90_rollout", 0)
        max_rollout = results.get("max_rollout", 0)

        print(
            f"| {config.model_name:<16} | {results['num_episodes']:>8} | {results['mean_eval_metric']:>10.2f} | {avg_achievements:>16.2f} | {total_unique:>19} | {shaped_reward:>13.3f} | {mean_k_score:>12.3f} | {q2_rollout:>6} | {p90_rollout:>7} | {max_rollout:>7} |"
        )

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Crafter ReAct Agent Evaluation")
    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--model", "-m", type=str, help="Model name (overrides config)")
    parser.add_argument("--episodes", "-e", type=int, help="Number of episodes (overrides config)")
    parser.add_argument(
        "--max-turns", "-t", type=int, help="Maximum turns per episode (overrides config)"
    )
    parser.add_argument("--difficulty", "-d", type=str, help="Difficulty level (overrides config)")

    # Custom OpenAI endpoint support
    parser.add_argument(
        "--openai-base-url",
        type=str,
        help="Custom OpenAI-compatible base URL (e.g., https://lora-service.modal.run)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="dummy",
        help="API key for custom endpoint (default: 'dummy')",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = CrafterConfig(args.config)
    else:
        # Try to load default config
        default_config_path = (
            Path(__file__).parent.parent.parent.parent / "evals" / "configs" / "crafter.toml"
        )
        if default_config_path.exists():
            config = CrafterConfig(str(default_config_path))
        else:
            config = CrafterConfig()

    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.episodes:
        config.num_instances = args.episodes
    if args.max_turns:
        config.max_turns = args.max_turns
    if args.difficulty:
        config.difficulty = args.difficulty

    # Configure custom OpenAI endpoint if provided
    if args.openai_base_url:
        config.set_custom_endpoint(args.openai_base_url, args.openai_api_key)

    # Fail fast if model_name still missing
    if not config.model_name:
        raise ValueError(
            "CrafterConfig: 'model_name' must be specified in the TOML config or via --model CLI argument; no fallback default."
        )

    asyncio.run(main())
