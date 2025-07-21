#!/usr/bin/env python3
"""
Test script to run ReAct agents against NetHack environment on synth service (port 8901)
Tests on multiple easy NetHack instances with enhanced debugging
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from httpx import AsyncClient
import sys
import os
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool


# --- Configuration Class ---
class NetHackConfig:
    """Configuration for NetHack evaluation (mirrors CrafterConfig)."""

    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.model_name = "gpt-4.1-mini"
        self.num_instances = 2
        self.max_turns = 40
        self.difficulty = "beginner"
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.save_traces = True
        self.save_detailed_results = True

        # Load from TOML if supplied
        if config_path and os.path.exists(config_path):
            try:
                import toml

                cfg = toml.load(config_path)

                eval_cfg = cfg.get("eval", {})
                self.model_name = eval_cfg.get("model_name", self.model_name)
                self.num_instances = eval_cfg.get("episodes", self.num_instances)
                self.max_turns = eval_cfg.get("max_steps", self.max_turns)
                self.difficulty = eval_cfg.get("difficulty", self.difficulty)
                self.seed = eval_cfg.get("seed", self.seed)

                svc_cfg = cfg.get("service", {})
                self.service_base_url = svc_cfg.get("base_url", self.service_base_url)
                self.service_timeout = svc_cfg.get("timeout", self.service_timeout)

                out_cfg = cfg.get("output", {})
                self.save_traces = out_cfg.get("save_traces", self.save_traces)
                self.save_detailed_results = out_cfg.get(
                    "save_detailed_results", self.save_detailed_results
                )
            except Exception as e:
                print(f"[WARNING] Failed to load config from {config_path}: {e}")


# Instantiate default config (may be overridden by CLI later)
config = NetHackConfig()


# Overwrite the original global constants to use config values (so rest of script works unchanged)
def _apply_config_to_globals(cfg: NetHackConfig):
    globals()["MODEL_NAME"] = cfg.model_name
    globals()["NUM_INSTANCES"] = cfg.num_instances
    globals()["MAX_TURNS"] = cfg.max_turns
    globals()["DIFFICULTY"] = cfg.difficulty
    globals()["SERVICE_BASE_URL"] = cfg.service_base_url


_apply_config_to_globals(config)

# --- CLI Override (similar to Crafter script) ---
# CLI parsing moved to end of file after main() is defined


# --- Service Configuration ---
SERVICE_BASE_URL = "http://localhost:8901"
MODEL_NAME = "gpt-4.1-mini"
NUM_INSTANCES = 2
MAX_TURNS = 40
DIFFICULTY = "beginner"  # beginner, beginner, intermediate, advanced, expert


# --- Tool Definitions ---
class NetHackActionArgs(BaseModel):
    """Arguments for nethack actions."""

    actions: List[str] = Field(
        description="List of 1-3 action names to execute in sequence (e.g., ['north', 'search', 'inventory'])"
    )
    reasoning: str = Field(description="Brief explanation of why these actions were chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


class NetHackActionTool(BaseTool):
    """Tool for performing actions in the NetHack environment."""

    name: str = "interact"
    arguments: type[BaseModel] = NetHackActionArgs
    description: str = "Perform 1-3 actions in sequence in the NetHack environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""

    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"

        # Define tools in OpenAI format
        self.tools = [
            NetHackActionTool(),
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
                    "actions": ["inventory"],
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


# --- NetHack ReAct Agent ---
class NetHackReActAgent(BaseReActAgent):
    """ReAct agent for NetHack environment."""

    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "nethack-react-agent"

    def get_system_message(self) -> str:
        return """You are an expert NetHack player. Your goal is to explore the dungeon, survive, and make progress.

MOVEMENT ACTIONS:
- north, south, east, west: Move in cardinal directions
- northeast, northwest, southeast, southwest: Move diagonally
- go_up, go_down: Use stairs (must be on < or > symbol)

EXPLORATION ACTIONS:
- search: Look for secret doors or traps
- open: Open doors
- close: Close doors
- look: Examine surroundings (FREE ACTION)

INVENTORY ACTIONS:
- inventory: Check your items (FREE ACTION)
- pickup: Pick up items
- drop: Drop items
- wear: Put on armor
- wield: Equip weapon
- eat: Consume food
- drink: Drink potion
- read: Read scroll

INTERACTION:
- wait: Rest for one turn
- chat: Talk to NPCs
- pay: Pay shopkeeper
- kick: Kick something

MAP SYMBOLS:
- @ = you (the player)
- . = floor
- # = wall/corridor
- + = closed door
- - = open door
- < = stairs up
- > = stairs down
- $ = gold
- % = food
- ! = potion
- ? = scroll
- / = wand
- ) = weapon
- [ = armor
- d,f = pets (dog/cat)
- Letters = monsters

STRATEGY:
1. Explore systematically to map the dungeon
2. Collect useful items and gold
3. Manage hunger by eating food
4. Fight weak monsters for experience
5. Use 'look' and 'inventory' frequently (they're free!)
6. Be cautious around unknown monsters

Remember: NetHack is complex but rewarding. Take your time and observe carefully."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for NetHack."""
        parts = []

        if "ascii_map" in obs:
            parts.append("ASCII Map:")
            parts.append(obs["ascii_map"])

        if "message" in obs and obs["message"]:
            parts.append(f"Message: {obs['message']}")

        if "character_stats" in obs:
            stats = obs["character_stats"]
            stat_items = []
            for key, value in stats.items():
                if key in ["HP", "level", "gold", "score", "turn"]:
                    stat_items.append(f"{key}: {value}")
            if stat_items:
                parts.append(f"Stats: {', '.join(stat_items)}")

        if "inventory_summary" in obs:
            parts.append(f"Inventory: {obs['inventory_summary']}")

        if "hunger_status" in obs and obs["hunger_status"]:
            parts.append(f"Hunger: {obs['hunger_status']}")

        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")

        if "reward" in obs:
            parts.append(f"Reward: {obs['reward']}")

        return "\n".join(parts) if parts else "No formatted observation available"


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient,
    agent: NetHackReActAgent,
    task_instance,
    instance_num: int,
    progress_bar=None,
) -> Dict[str, Any]:
    """Run a single NetHack episode and return episode metrics."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/NetHack/initialize", json={"task_instance": await task_instance.serialize()}
        )

        if create_resp.status_code != 200:
            print(
                f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}"
            )
            return {"eval_metric": 0.0, "rubric": {}, "error": True}

        env_id = create_resp.json()["env_id"]

        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)

        # DEBUG: Print initial state
        # print(f"\n  Instance {instance_num}: Starting NetHack adventure")
        # print(f"  Character: {task_instance.metadata.character_role}")
        # print(f"  Goal: Reach depth {task_instance.metadata.target_depth}")

        # Track progress
        initial_depth = 1
        max_depth_reached = initial_depth
        max_reward = 0.0
        final_stats = {}
        balrog_score = 0.0
        balrog_total_reward = 0.0
        achievements_unlocked = []

        # Track additional progress metrics
        monsters_killed = 0
        items_picked_up = 0
        scrolls_read = 0
        potions_drunk = 0
        rooms_explored = 0
        secret_doors_found = 0
        stairs_found = 0
        traps_encountered = 0
        spells_cast = 0
        prayers_attempted = 0
        max_score = 0

        # Track shaped rewards (requires previous observation)
        prev_obs = None
        shaped_rewards = {
            # Survival & Progress
            "depth_delta_total": 0.0,
            "stairs_seen_total": 0,
            "turn_alive_total": 0.0,
            "hp_gain_total": 0.0,
            "hunger_ok_total": 0,
            # Exploration
            "new_tiles_total": 0,
            "rooms_explored_delta_total": 0,
            "secret_doors_delta_total": 0,
            "traps_identified_delta_total": 0,
            # Combat
            "monsters_killed_delta_total": 0,
            "dmg_dealt_total": 0.0,
            "dmg_taken_total": 0.0,
            # Resources
            "gold_delta_total": 0,
            "items_picked_delta_total": 0,
            "scrolls_read_delta_total": 0,
            "potions_quaffed_delta_total": 0,
            "spells_cast_delta_total": 0,
            # Skill/Utility
            "first_prayer_achieved": False,
            "first_spell_achieved": False,
            "identify_item_total": 0,
            # Achievements
            "achievement_unlocked_total": 0,
            # Intermediate reward accumulation
            "total_intermediate_reward": 0.0,
        }

        # Run episode
        for turn in range(agent.max_turns):
            # Get agent decision
            action = await agent.decide(formatted_obs, agent.get_system_message(), turn)

            # Check for termination
            if action["name"] == "terminate":
                print(
                    f"  Agent terminated: {action['parameters'].get('reason', 'no reason given')}"
                )
                break

            # Execute actions in environment
            action_sequence = action["parameters"]["actions"]

            step_resp = await client.post(
                f"/env/NetHack/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"actions": action_sequence}}]
                    },
                },
            )

            if step_resp.status_code != 200:
                print(f"  ❌ Step failed: {step_resp.status_code}: {step_resp.text}")
                break

            obs = step_resp.json()["observation"]
            formatted_obs = agent.format_observation(obs)

            # Calculate shaped rewards if we have a previous observation
            if prev_obs is not None:
                # --- Survival & Progress ---
                current_depth = obs.get("character_stats", {}).get("dungeon_level", 1)
                prev_depth = prev_obs.get("character_stats", {}).get("dungeon_level", 1)
                depth_delta = current_depth - prev_depth
                shaped_rewards["depth_delta_total"] += depth_delta

                stairs_seen = int(obs.get("stairs_found", 0) > prev_obs.get("stairs_found", 0))
                shaped_rewards["stairs_seen_total"] += stairs_seen

                shaped_rewards["turn_alive_total"] += 0.01  # tiny tick reward every step survived

                # HP calculations
                current_hp = obs.get("character_stats", {}).get("hp", 1)
                current_max_hp = obs.get("character_stats", {}).get("max_hp", 1)
                prev_hp = prev_obs.get("character_stats", {}).get("hp", 1)
                prev_max_hp = prev_obs.get("character_stats", {}).get("max_hp", 1)

                if current_max_hp > 0 and prev_max_hp > 0:
                    hp_pct = current_hp / current_max_hp
                    prev_hp_pct = prev_hp / prev_max_hp
                    hp_gain = hp_pct - prev_hp_pct
                    shaped_rewards["hp_gain_total"] += hp_gain

                hunger_ok = int(obs.get("hunger_status", "") in ("Not hungry", "Satiated", ""))
                shaped_rewards["hunger_ok_total"] += hunger_ok

                # --- Exploration ---
                new_tiles = obs.get("exploration_stats", {}).get("new_tiles", 0)
                shaped_rewards["new_tiles_total"] += new_tiles

                rooms_explored_delta = obs.get("rooms_explored", 0) - prev_obs.get(
                    "rooms_explored", 0
                )
                shaped_rewards["rooms_explored_delta_total"] += rooms_explored_delta

                secret_doors_delta = obs.get("secret_doors_found", 0) - prev_obs.get(
                    "secret_doors_found", 0
                )
                shaped_rewards["secret_doors_delta_total"] += secret_doors_delta

                traps_identified_delta = obs.get("traps_encountered", 0) - prev_obs.get(
                    "traps_encountered", 0
                )
                shaped_rewards["traps_identified_delta_total"] += traps_identified_delta

                # --- Combat ---
                monsters_killed_delta = obs.get("achievement_stats", {}).get(
                    "monsters_killed", 0
                ) - prev_obs.get("achievement_stats", {}).get("monsters_killed", 0)
                shaped_rewards["monsters_killed_delta_total"] += monsters_killed_delta

                dmg_dealt = obs.get("combat", {}).get("damage_dealt", 0)
                shaped_rewards["dmg_dealt_total"] += dmg_dealt

                dmg_taken = obs.get("combat", {}).get("damage_taken", 0)
                shaped_rewards["dmg_taken_total"] += dmg_taken

                # --- Resources ---
                gold_delta = obs.get("character_stats", {}).get("gold", 0) - prev_obs.get(
                    "character_stats", {}
                ).get("gold", 0)
                shaped_rewards["gold_delta_total"] += gold_delta

                items_picked_delta = obs.get("items_collected", 0) - prev_obs.get(
                    "items_collected", 0
                )
                shaped_rewards["items_picked_delta_total"] += items_picked_delta

                scrolls_read_delta = obs.get("scrolls_read", 0) - prev_obs.get("scrolls_read", 0)
                shaped_rewards["scrolls_read_delta_total"] += scrolls_read_delta

                potions_quaffed_delta = obs.get("potions_drunk", 0) - prev_obs.get(
                    "potions_drunk", 0
                )
                shaped_rewards["potions_quaffed_delta_total"] += potions_quaffed_delta

                spells_cast_delta = obs.get("spells_cast", 0) - prev_obs.get("spells_cast", 0)
                shaped_rewards["spells_cast_delta_total"] += spells_cast_delta

                # --- Skill/Utility ---
                if (
                    obs.get("prayers_attempted", 0) > 0
                    and prev_obs.get("prayers_attempted", 0) == 0
                ):
                    shaped_rewards["first_prayer_achieved"] = True

                if spells_cast_delta > 0 and prev_obs.get("spells_cast", 0) == 0:
                    shaped_rewards["first_spell_achieved"] = True

                message = obs.get("message", "")
                if isinstance(message, bytes):
                    message = message.decode("ascii", errors="ignore").strip("\x00")
                if "You identify" in message:
                    shaped_rewards["identify_item_total"] += 1

                # --- Achievements ---
                current_achievements = obs.get("achievements_unlocked", {})
                prev_achievements = prev_obs.get("achievements_unlocked", {})
                achievement_unlocked = sum(
                    int(v and not prev_achievements.get(k, False))
                    for k, v in current_achievements.items()
                )
                shaped_rewards["achievement_unlocked_total"] += achievement_unlocked

                # --- Calculate intermediate reward ---
                intermediate_reward = (
                    1.0 * depth_delta
                    + 0.2 * new_tiles
                    + 2.0 * monsters_killed_delta
                    - 0.5 * dmg_taken / 10
                    + 0.1 * gold_delta
                    + 5.0 * achievement_unlocked
                )
                shaped_rewards["total_intermediate_reward"] += intermediate_reward

            # Store current observation as previous for next iteration
            prev_obs = obs.copy() if obs else None

            # Track progress
            if "character_stats" in obs:
                final_stats = obs["character_stats"]
                if "dungeon_level" in final_stats:
                    current_depth = final_stats["dungeon_level"]
                    max_depth_reached = max(max_depth_reached, current_depth)

            reward = obs.get("reward", 0.0)
            max_reward = max(max_reward, reward)

            # Track achievements and Balrog rewards (like in main agent)
            if "achievements_unlocked" in obs:
                for ach, unlocked in obs["achievements_unlocked"].items():
                    if unlocked and ach not in achievements_unlocked:
                        achievements_unlocked.append(ach)

            if "balrog_total_reward" in obs:
                balrog_total_reward = obs["balrog_total_reward"]

            if "achievement_stats" in obs and "balrog_score" in obs["achievement_stats"]:
                balrog_score = obs["achievement_stats"]["balrog_score"]

            # Track additional progress metrics from achievement stats
            if "achievement_stats" in obs:
                ach_stats = obs["achievement_stats"]
                monsters_killed = ach_stats.get("monsters_killed", 0)
                items_picked_up = ach_stats.get("items_collected", 0)
                rooms_explored = ach_stats.get("rooms_explored", 0)
                secret_doors_found = ach_stats.get("secret_doors_found", 0)
                stairs_found = ach_stats.get("stairs_found", 0)

            # Track score progression
            current_score = obs.get("score", 0)
            max_score = max(max_score, current_score)

            # Parse message for additional events
            message = obs.get("message", "")
            if isinstance(message, bytes):
                message = message.decode("ascii", errors="ignore").strip("\x00")

            # Look for specific events in messages
            if "You read" in message:
                scrolls_read += 1
            elif "You drink" in message:
                potions_drunk += 1
            elif "You cast" in message or "spell" in message.lower():
                spells_cast += 1
            elif "You pray" in message:
                prayers_attempted += 1
            elif "trap" in message.lower():
                traps_encountered += 1

            # Check if episode ended
            terminated = obs.get("terminated", False)

            if terminated:
                print(
                    f"  📊 Instance {instance_num}: Episode ended at depth {max_depth_reached}, reward: {max_reward:.3f}"
                )
                break

            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)

        # Cleanup
        await client.post(f"/env/NetHack/terminate", json={"env_id": env_id})

        # Ensure progress bar completes
        if progress_bar is not None:
            progress_bar.n = progress_bar.total
            progress_bar.close()

        # Calculate eval metric and rubric
        target_depth = task_instance.metadata.target_depth

        # Balrog score: Use proper score from observation (like in main agent)
        # This is the standard NetHack evaluation metric

        # Eval metric is the normalized Balrog score (0-1)
        eval_metric = balrog_score / 100.0

        # Create rubric with specific achievements
        rubric = {
            # Core progression metrics
            "reached_target_depth": 1.0 if max_depth_reached >= target_depth else 0.0,
            "depth_progress": min(1.0, max_depth_reached / target_depth),
            "gained_experience": 1.0 if final_stats.get("experience", 0) > 0 else 0.0,
            "collected_gold": 1.0 if final_stats.get("gold", 0) > 100 else 0.0,
            "gained_levels": 1.0 if final_stats.get("level", 1) > 1 else 0.0,
            "survived_turns": min(1.0, len(agent.history) / 20.0),  # Normalize to 20 turns
            "positive_reward": 1.0 if max_reward > 0 else 0.0,
            "achievement_fraction": len(achievements_unlocked)
            / 100.0,  # Core Balrog metric (approximated)
            # Combat and interaction metrics
            "monsters_defeated": min(1.0, monsters_killed / 5.0),  # Normalize to 5 kills
            "items_collected": min(1.0, items_picked_up / 10.0),  # Normalize to 10 items
            "scrolls_used": min(1.0, scrolls_read / 3.0),  # Normalize to 3 scrolls
            "potions_used": min(1.0, potions_drunk / 2.0),  # Normalize to 2 potions
            "spells_cast": min(1.0, spells_cast / 2.0),  # Normalize to 2 spells
            # Exploration metrics
            "rooms_explored": min(1.0, rooms_explored / 5.0),  # Normalize to 5 rooms
            "secret_doors_found": 1.0 if secret_doors_found > 0 else 0.0,
            "stairs_found": 1.0 if stairs_found > 0 else 0.0,
            "traps_encountered": 1.0 if traps_encountered > 0 else 0.0,
            # Advanced metrics
            "prayers_attempted": 1.0 if prayers_attempted > 0 else 0.0,
            "score_progress": min(1.0, max_score / 100.0),  # Normalize to 100 points
            "active_exploration": 1.0
            if (rooms_explored + secret_doors_found + stairs_found) > 0
            else 0.0,
            "item_interaction": 1.0 if (scrolls_read + potions_drunk + spells_cast) > 0 else 0.0,
            # --- Shaped Rewards ---
            # Survival & Progress
            "depth_progress_reward": max(0.0, shaped_rewards["depth_delta_total"]),
            "stairs_discovery_reward": min(1.0, shaped_rewards["stairs_seen_total"] / 5.0),
            "survival_reward": min(
                1.0, shaped_rewards["turn_alive_total"] / 1.0
            ),  # Normalize to 1.0 for 100 turns
            "hp_management_reward": max(0.0, shaped_rewards["hp_gain_total"]),
            "hunger_management_reward": min(
                1.0, shaped_rewards["hunger_ok_total"] / (len(agent.history) or 1)
            ),
            # Exploration
            "new_tiles_reward": min(
                1.0, shaped_rewards["new_tiles_total"] / 100.0
            ),  # Normalize to 100 tiles
            "room_discovery_reward": min(1.0, shaped_rewards["rooms_explored_delta_total"] / 5.0),
            "secret_discovery_reward": min(1.0, shaped_rewards["secret_doors_delta_total"] / 3.0),
            "trap_discovery_reward": min(1.0, shaped_rewards["traps_identified_delta_total"] / 3.0),
            # Combat
            "combat_success_reward": min(1.0, shaped_rewards["monsters_killed_delta_total"] / 5.0),
            "damage_dealt_reward": min(1.0, shaped_rewards["dmg_dealt_total"] / 50.0),
            "damage_avoided_reward": max(0.0, 1.0 - shaped_rewards["dmg_taken_total"] / 50.0),
            # Resources
            "wealth_accumulation_reward": min(1.0, shaped_rewards["gold_delta_total"] / 100.0),
            "item_collection_reward": min(1.0, shaped_rewards["items_picked_delta_total"] / 10.0),
            "scroll_usage_reward": min(1.0, shaped_rewards["scrolls_read_delta_total"] / 3.0),
            "potion_usage_reward": min(1.0, shaped_rewards["potions_quaffed_delta_total"] / 3.0),
            "spell_usage_reward": min(1.0, shaped_rewards["spells_cast_delta_total"] / 3.0),
            # Skill/Utility
            "first_prayer_reward": 1.0 if shaped_rewards["first_prayer_achieved"] else 0.0,
            "first_spell_reward": 1.0 if shaped_rewards["first_spell_achieved"] else 0.0,
            "identification_reward": min(1.0, shaped_rewards["identify_item_total"] / 3.0),
            # Achievements
            "achievement_unlock_reward": min(
                1.0, shaped_rewards["achievement_unlocked_total"] / 10.0
            ),
            # Overall shaped reward
            "total_intermediate_reward": shaped_rewards["total_intermediate_reward"],
            "normalized_intermediate_reward": min(
                1.0, max(0.0, shaped_rewards["total_intermediate_reward"] / 20.0)
            ),
        }

        # Remove or mark irrelevant rubric keys
        irrelevant_rubric = {}
        for k in list(rubric.keys()):
            if k in IRRELEVANT_RUBRIC_KEYS:
                irrelevant_rubric[k] = rubric.pop(k)

        # Success determination
        success = max_depth_reached >= target_depth or max_reward > 10.0 or balrog_score > 5.0

        if success:
            print(
                f"  ✅ Instance {instance_num}: SUCCESS! Depth {max_depth_reached}, Balrog score: {balrog_score:.0f}"
            )
        else:
            print(
                f"  ❌ Instance {instance_num}: Partial progress - depth {max_depth_reached}/{target_depth}, Balrog score: {balrog_score:.0f}"
            )

        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "max_depth_reached": max_depth_reached,
            "target_depth": target_depth,
            "max_reward": max_reward,
            "balrog_score": balrog_score,
            "balrog_total_reward": balrog_total_reward,
            "achievements_unlocked": achievements_unlocked,
            "final_stats": final_stats,
            "success": success,
            "error": False,
            # Additional progress metrics
            "monsters_killed": monsters_killed,
            "items_picked_up": items_picked_up,
            "scrolls_read": scrolls_read,
            "potions_drunk": potions_drunk,
            "rooms_explored": rooms_explored,
            "secret_doors_found": secret_doors_found,
            "stairs_found": stairs_found,
            "traps_encountered": traps_encountered,
            "spells_cast": spells_cast,
            "prayers_attempted": prayers_attempted,
            "max_score": max_score,
            # Shaped rewards
            "shaped_rewards": shaped_rewards,
            "irrelevant_rubric": irrelevant_rubric,
        }

    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback

        traceback.print_exc()
        return {"eval_metric": 0.0, "rubric": {}, "error": True}


# --- Batch Evaluation ---
async def evaluate_nethack_batch() -> Dict[str, Any]:
    """Evaluate NetHack agent on multiple easy instances."""
    print(f"🎯 Evaluating NetHack on {NUM_INSTANCES} {DIFFICULTY} instances...")

    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)

    # Get task instances using the taskset system
    from synth_ai.environments.examples.nethack.taskset import create_nethack_taskset

    taskset = await create_nethack_taskset()

    # Filter for the desired difficulty
    task_instances = [inst for inst in taskset.instances if inst.metadata.difficulty == DIFFICULTY][
        :NUM_INSTANCES
    ]

    if len(task_instances) < NUM_INSTANCES:
        print(f"  ⚠️  Only found {len(task_instances)} {DIFFICULTY} instances, using all available")

    print(f"  📝 Using {len(task_instances)} {DIFFICULTY} task instances")

    async with AsyncClient(
        base_url=SERVICE_BASE_URL, timeout=60.0
    ) as client:  # Longer timeout for NetHack
        tasks = []
        bars = []
        for i, task_instance in enumerate(task_instances):
            bar = tqdm(total=MAX_TURNS, desc=f"Ep {i + 1}", position=i, leave=True)
            bars.append(bar)
            agent = NetHackReActAgent(llm, max_turns=MAX_TURNS, verbose=False)
            tasks.append(run_single_episode(client, agent, task_instance, i + 1, bar))

        results = await asyncio.gather(*tasks)

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

        # Extract Balrog scores
        balrog_scores = [r.get("balrog_score", 0.0) for r in valid_results]
        mean_balrog_score = sum(balrog_scores) / len(balrog_scores) if balrog_scores else 0.0

        # Extract Balrog total rewards
        balrog_total_rewards = [r.get("balrog_total_reward", 0.0) for r in valid_results]
        mean_balrog_total_reward = (
            sum(balrog_total_rewards) / len(balrog_total_rewards) if balrog_total_rewards else 0.0
        )

        # Extract additional progress metrics
        progress_metrics = {
            "monsters_killed": [r.get("monsters_killed", 0) for r in valid_results],
            "items_picked_up": [r.get("items_picked_up", 0) for r in valid_results],
            "scrolls_read": [r.get("scrolls_read", 0) for r in valid_results],
            "potions_drunk": [r.get("potions_drunk", 0) for r in valid_results],
            "rooms_explored": [r.get("rooms_explored", 0) for r in valid_results],
            "secret_doors_found": [r.get("secret_doors_found", 0) for r in valid_results],
            "stairs_found": [r.get("stairs_found", 0) for r in valid_results],
            "traps_encountered": [r.get("traps_encountered", 0) for r in valid_results],
            "spells_cast": [r.get("spells_cast", 0) for r in valid_results],
            "prayers_attempted": [r.get("prayers_attempted", 0) for r in valid_results],
            "max_score": [r.get("max_score", 0) for r in valid_results],
        }

        # Calculate means for progress metrics
        mean_progress_metrics = {}
        for key, values in progress_metrics.items():
            mean_progress_metrics[key] = sum(values) / len(values) if values else 0.0

        # Extract shaped rewards
        shaped_rewards_summary = {}
        irrelevant_shaped_summary = {}
        if valid_results and "shaped_rewards" in valid_results[0]:
            shaped_reward_keys = valid_results[0]["shaped_rewards"].keys()
            for key in shaped_reward_keys:
                values = [r.get("shaped_rewards", {}).get(key, 0) for r in valid_results]
                if isinstance(values[0], bool):
                    avg_value = sum(values) / len(values)  # Fraction of episodes
                else:
                    avg_value = sum(values) / len(values) if values else 0.0

                if key in IRRELEVANT_RUBRIC_KEYS:
                    irrelevant_shaped_summary[key] = avg_value
                else:
                    shaped_rewards_summary[key] = avg_value

        # Calculate individual relevant shaped rewards sums
        individual_relevant_sums = []
        if valid_results and "shaped_rewards" in valid_results[0]:
            for result in valid_results:
                episode_shaped_rewards = result.get("shaped_rewards", {})
                relevant_sum = sum(
                    v for k, v in episode_shaped_rewards.items() if k not in IRRELEVANT_RUBRIC_KEYS
                )
                individual_relevant_sums.append(relevant_sum)

        # Calculate mean of relevant shaped rewards sums
        relevant_shaped_rewards_sum = (
            sum(individual_relevant_sums) / len(individual_relevant_sums)
            if individual_relevant_sums
            else 0.0
        )

        # Calculate individual relevant rubric sums
        individual_relevant_rubric_sums = []
        for result in valid_results:
            episode_rubric = result.get("rubric", {})
            relevant_rubric_sum = sum(
                v for k, v in episode_rubric.items() if k not in IRRELEVANT_RUBRIC_KEYS
            )
            individual_relevant_rubric_sums.append(relevant_rubric_sum)

        # Calculate mean of relevant rubric sums
        relevant_rubric_sum = (
            sum(individual_relevant_rubric_sums) / len(individual_relevant_rubric_sums)
            if individual_relevant_rubric_sums
            else 0.0
        )

        # Calculate mean rubric values (excluding irrelevant)
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(
                [k for k in r["rubric"].keys() if k not in IRRELEVANT_RUBRIC_KEYS]
            )

        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)

        # Collect irrelevant rubric metrics summary
        irrelevant_summary = {}
        for key in IRRELEVANT_RUBRIC_KEYS:
            vals = [r.get("irrelevant_rubric", {}).get(key, 0.0) for r in valid_results]
            irrelevant_summary[key] = sum(vals) / len(vals) if vals else 0.0

        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "balrog_scores": balrog_scores,
            "mean_balrog_score": mean_balrog_score,
            "balrog_total_rewards": balrog_total_rewards,
            "mean_balrog_total_reward": mean_balrog_total_reward,
            "mean_rubric": mean_rubric,
            "progress_metrics": progress_metrics,
            "mean_progress_metrics": mean_progress_metrics,
            "shaped_rewards_summary": shaped_rewards_summary,
            "irrelevant_summary": irrelevant_summary,
            "irrelevant_shaped_summary": irrelevant_shaped_summary,
            "relevant_shaped_rewards_sum": relevant_shaped_rewards_sum,
            "individual_relevant_sums": individual_relevant_sums,
            "individual_relevant_rubric_sums": individual_relevant_rubric_sums,
            "relevant_rubric_sum": relevant_rubric_sum,
            "num_episodes": len(valid_results),
        }


async def main():
    """Run NetHack evaluation."""
    print(f"🎮 NetHack ReAct Agent Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Service: {SERVICE_BASE_URL}")
    print(f"Instances: {NUM_INSTANCES}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Test service health
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=10.0) as client:
        try:
            health_resp = await client.get("/health")
            health_data = health_resp.json()

            if "NetHack" not in health_data.get("supported_environments", []):
                print("❌ NetHack not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return

    # Run evaluation
    try:
        results = await evaluate_nethack_batch()

        print("\n" + "=" * 80)
        print("🏆 FINAL NETHACK EVALUATION RESULTS")
        print("=" * 80)

        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.2f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")

        # Print Balrog scores
        print(f"\n⚔️  BALROG SCORES:")
        print(f"  Individual Scores: {[f'{x:.3f}' for x in results['balrog_scores']]}")
        print(f"  Mean Balrog Score: {results['mean_balrog_score']:.3f}")

        # Print Balrog total rewards
        print(f"\n🏆 BALROG TOTAL REWARDS:")
        print(f"  Individual Rewards: {[f'{x:.2f}' for x in results['balrog_total_rewards']]}")
        print(f"  Mean Balrog Total Reward: {results['mean_balrog_total_reward']:.2f}")

        # Print relevant sums
        print(f"\n💯 RELEVANT RUBRIC SUMS:")
        print(
            f"  Individual Sums: {[f'{x:.3f}' for x in results.get('individual_relevant_rubric_sums', [])]}"
        )
        print(f"  Mean Relevant Rubric Sum: {results.get('relevant_rubric_sum', 0.0):.3f}")

        print(f"\n💯 RELEVANT SHAPED REWARD SUMS:")
        print(
            f"  Individual Sums: {[f'{x:.3f}' for x in results.get('individual_relevant_sums', [])]}"
        )
        print(
            f"  Mean Relevant Shaped Reward Sum: {results.get('relevant_shaped_rewards_sum', 0.0):.3f}"
        )

        # Print rubric results
        print(f"\n🎯 RUBRIC RESULTS:")
        if results["mean_rubric"]:
            for achievement, score in sorted(results["mean_rubric"].items()):
                print(f"  {achievement}: {score:.2f}")
        else:
            print("  No rubric data available")

        # Print progress metrics
        print(f"\n📈 PROGRESS METRICS:")
        if results["mean_progress_metrics"]:
            for metric, value in sorted(results["mean_progress_metrics"].items()):
                print(f"  {metric}: {value:.1f}")
        else:
            print("  No progress data available")

        # Print shaped rewards summary
        print(f"\n🎯 SHAPED REWARDS SUMMARY:")
        if results.get("shaped_rewards_summary"):
            for reward_key, value in sorted(results["shaped_rewards_summary"].items()):
                if isinstance(value, bool):
                    print(f"  {reward_key}: {value}")
                else:
                    print(f"  {reward_key}: {value:.3f}")
        else:
            print("  No shaped rewards data available")

        # Print irrelevant shaped rewards
        print(f"\n🚫 IRRELEVANT SHAPED REWARDS:")
        if results.get("irrelevant_shaped_summary"):
            for reward_key, value in sorted(results["irrelevant_shaped_summary"].items()):
                print(f"  {reward_key}: {value:.3f}")
        else:
            print("  None")

        # Print irrelevant rubric metrics
        print(f"\n🚫 IRRELEVANT RUBRIC METRICS:")
        if results.get("irrelevant_summary"):
            for metric, value in sorted(results["irrelevant_summary"].items()):
                print(f"  {metric}: {value:.2f}")
        else:
            print("  None")

        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        balrog_score = results["mean_balrog_score"]
        eval_metric = results["mean_eval_metric"]

        if eval_metric > 0.8 or balrog_score > 40.0:
            print("🎉 Excellent performance - mastering the dungeon!")
        elif eval_metric > 0.6 or balrog_score > 20.0:
            print("✅ Good performance - making solid progress!")
        elif eval_metric > 0.4 or balrog_score > 10.0:
            print("⚠️  Moderate performance - learning the ropes")
        elif balrog_score > 5.0:
            print("📈 Decent exploration - building dungeon skills")
        else:
            print("🏃 Early exploration - focus on basic survival and movement")

        # Output markdown table row for README collation
        print(f"\n📋 MARKDOWN TABLE ROW:")
        print(
            "| Model            | Episodes | Mean Eval | Mean Balrog | Mean Relevant Rubric | Mean Relevant Shaped | Non-Zero Progress | Non-Zero Rubric | Assessment |"
        )
        print(
            "|------------------|----------|-----------|-------------|----------------------|----------------------|-------------------|-----------------|------------|"
        )
        relevant_rubric_sum = results.get("relevant_rubric_sum", 0.0)
        relevant_shaped_sum = results.get("relevant_shaped_rewards_sum", 0.0)

        # Count non-zero progress metrics
        progress_metrics = results.get("mean_progress_metrics", {})
        non_zero_progress = sum(1 for value in progress_metrics.values() if value > 0.0)

        # Count non-zero rubric results (excluding irrelevant ones)
        rubric_results = results.get("mean_rubric", {})
        non_zero_rubric = sum(
            1
            for key, value in rubric_results.items()
            if value > 0.0 and key not in IRRELEVANT_RUBRIC_KEYS
        )

        if eval_metric > 0.6 or balrog_score > 20.0:
            assessment = "Excellent"
        elif eval_metric > 0.4 or balrog_score > 10.0:
            assessment = "Good"
        elif balrog_score > 5.0:
            assessment = "Moderate"
        else:
            assessment = "Learning"

        print(
            f"| {MODEL_NAME:<16} | {results['num_episodes']:>8} | {eval_metric:>9.3f} | {balrog_score:>11.3f} | {relevant_rubric_sum:>20.3f} | {relevant_shaped_sum:>20.3f} | {non_zero_progress:>17} | {non_zero_rubric:>15} | {assessment:<10} |"
        )

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


# Metrics that are considered baseline / always-positive and should be treated as irrelevant when summarizing
IRRELEVANT_RUBRIC_KEYS = {
    "survival_reward",
    "hunger_management_reward",
    "damage_avoided_reward",
    "stairs_discovery_reward",
    "turn_alive_total",  # from shaped summary
    "hunger_ok_total",  # from shaped summary
}

# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Run NetHack ReAct Agent Evaluation (TOML configurable)"
    )
    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--model", "-m", type=str, help="Model name (overrides config)")
    parser.add_argument("--episodes", "-e", type=int, help="Number of episodes (overrides config)")
    parser.add_argument("--max-turns", "-t", type=int, help="Maximum turns (overrides config)")
    parser.add_argument("--difficulty", "-d", type=str, help="Difficulty (overrides config)")

    args = parser.parse_args()

    if args.config:
        config = NetHackConfig(args.config)
    else:
        config = NetHackConfig()

    # Apply CLI overrides
    if args.model:
        config.model_name = args.model
    if args.episodes:
        config.num_instances = args.episodes
    if args.max_turns:
        config.max_turns = args.max_turns
    if args.difficulty:
        config.difficulty = args.difficulty

    _apply_config_to_globals(config)

    # Run the evaluation
    asyncio.run(main())
