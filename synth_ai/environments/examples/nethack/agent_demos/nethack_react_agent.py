"""ReAct agent demo for NetHack environment."""

import argparse
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field
from synth_ai.environments.examples.nethack.environment import NetHackEnvironment
from synth_ai.environments.examples.nethack.helpers import (
    extract_game_context,
    format_observation_for_llm,
    get_actions_for_context,
)
from synth_ai.environments.examples.nethack.taskset import (
    NetHackTaskInstanceMetadata,
    create_nethack_taskset,
)
from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str


class NetHackInteractArgs(BaseModel):
    """Arguments for NetHack interaction."""

    reasoning: str = Field(description="Explain your reasoning for these actions")
    actions: List[str] = Field(description="List of actions to perform in sequence")


class NetHackInteractTool(BaseTool):
    """Tool for interacting with NetHack environment"""

    name: str = "nethack_interact"
    arguments: type[BaseModel] = NetHackInteractArgs
    description: str = (
        "Perform one or more actions in NetHack. Use EXACT action names from the system prompt!\n"
        "\n"
        "VALID ACTIONS: north, south, east, west, northeast, northwest, southeast, southwest, "
        "wait, go_up, go_down, search, open, close, inventory, pickup, drop, wear, wield, "
        "eat, drink, read, zap, apply, throw, fire, cast, pray, look, help, quit, save, "
        "plus menu letters (a-z) and numbers (0-9).\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Use exact names: 'go_up' not 'up', 'open' not 'open door'\n"
        "- No compound actions: use ['open', 'west'] not ['open west']\n"
        "- Combat: move INTO monsters, there is no 'fight' action\n"
        "- Stairs: must be ON stairs (< or >) to use go_up/go_down\n"
        "\n"
        "Invalid actions will return helpful error messages."
    )


class TerminateTool(BaseTool):
    """Tool for terminating the game"""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the game when you die, complete objectives, or decide to quit"


class NetHackReActAgent:
    """ReAct agent for playing NetHack."""

    def __init__(self, llm: LM, max_turns: int = 50):
        self.llm = llm
        self.max_turns = max_turns
        self.history: List[Dict[str, Any]] = []
        self.recent_actions: List[Dict[str, Any]] = []  # Track recent tool calls
        self.system_name = "nethack-react"

        # Define tools using BaseTool classes
        self.tools = [NetHackInteractTool(), TerminateTool()]

    def _create_system_prompt(self, task_instructions: str) -> str:
        """Create the system prompt for the agent."""
        base_prompt = (
            """You are an expert NetHack player. Your goal is to navigate the dungeon and complete objectives.

"""
            + task_instructions
            + """

=== EXACT VALID ACTIONS ===
You MUST use only these exact action names. Invalid actions will cause errors!

MOVEMENT (consumes time):
- north, south, east, west, northeast, northwest, southeast, southwest
- wait (rest for one turn)
- go_up (use stairs up), go_down (use stairs down)
- run_north, run_south, run_east, run_west (run until something interesting)

EXPLORATION (consumes time):
- search (for secret doors/traps)
- open (open nearest door)
- close (close nearest door)  
- kick (kick something)
- force (force a lock)
- untrap (disarm trap)

INVENTORY (consumes time):
- inventory (check items - but this is actually FREE!)
- pickup (pick up items here)
- drop (drop items)
- dropall (drop everything)
- wear (put on armor)
- take_off (remove armor)
- wield (equip weapon)
- unwield (unequip weapon)
- quiver (ready ammunition)
- put_on (wear accessories)
- remove (take off accessories)

USING ITEMS (consumes time):
- eat (consume food)
- drink (drink potion - but NetHack calls this "quaff")
- read (read scroll/book)
- zap (use wand)
- apply (use tool)
- invoke (use artifact power)
- rub (rub lamp/stone)
- throw (throw item)
- fire (shoot from quiver)

MAGIC & DIVINE (consumes time):
- cast (cast spell)
- pray (pray to deity)
- offer (sacrifice at altar)
- turn_undead (priest ability)

CHARACTER ACTIONS (consumes time):
- enhance (improve skills)
- sit (sit down)
- pay (pay shopkeeper)
- chat (talk to someone)
- loot (search container)
- engrave (write on ground)

FREE ACTIONS (do NOT consume time):
- look (look around - FREE!)
- farlook (examine specific location - FREE!)
- whatis (identify map symbol - FREE!)
- identify (check item details - FREE!)
- discoveries (list known items - FREE!)
- conduct (check game conduct - FREE!)
- attributes (check character stats - FREE!)
- help (show help - FREE!)
- version (show version - FREE!)
- history (show message history - FREE!)

GAME COMMANDS:
- save (save game)
- quit (quit game)

MENU/PROMPT RESPONSES:
- yes, no, all, none, escape
- Single letters: a, b, c, ..., z, A, B, C, ..., Z
- Single numbers: 0, 1, 2, ..., 9

=== CRITICAL RULES ===

1. EXACT ACTION NAMES: Use only the exact names above! "go_up" not "up", "open" not "open door"

2. NO COMPOUND ACTIONS: Don't use "open west" or "go to stairs". Use separate actions: ["open", "west"]

3. COMBAT: NO "fight" action exists! To attack monsters, MOVE INTO them using directional moves!

4. STAIRS: Must be standing ON stairs (< or >) to use go_up/go_down. Move to them first!

5. FREE ACTIONS: look, inventory, help, etc. don't advance time. Use sparingly!

6. DOORS: Use "open" (not "open door"). It opens the nearest door automatically.

7. TIME MANAGEMENT: Every action except FREE ACTIONS consumes a turn and lets monsters act!

=== MAP SYMBOLS ===
@ = you (player)
f = kitten (pet), d = dog (pet)
a-z = monsters (letters are monsters!)
$ = gold, * = gem
+ = closed door, - = horizontal open door, | = vertical open door  
< = stairs up, > = stairs down
. = floor, # = corridor/wall
! = potion, ? = scroll, / = wand, = = ring, " = amulet
[ = armor, ) = weapon, ( = tool, % = food
^ = trap, { = fountain, } = pool

=== STRATEGY ===
1. EXPLORE: Move around systematically to map the dungeon
2. DESCEND: Find stairs down (>) and use go_down to reach deeper levels
3. COMBAT: Move into monsters to attack them
4. LOOT: Pick up gold ($) and useful items
5. SURVIVE: Monitor health, eat when hungry
6. ACHIEVE: Try to unlock achievements through exploration and combat

=== LEARNING FROM FEEDBACK ===
- If "No stairs here to up/down" → You're not standing on stairs, move to them first
- If "Unknown action" → You used an invalid action name, use exact names from list above
- If nothing happens after "look" → Use a time-consuming action to advance the game
- If "You see no objects here" → Move to explore other areas

Always think step by step and use the exact action names provided above!"""
        )

        return base_prompt

    def _update_last_action_result(self, result_message: str):
        """Update the result of the most recent action."""
        if self.recent_actions:
            self.recent_actions[-1]["result"] = result_message[:100]  # Truncate long messages

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM."""
        # Use our formatting utility
        formatted = format_observation_for_llm(obs)

        # Add recent actions context if we have any
        if self.recent_actions:
            formatted += "\n\n=== RECENT ACTIONS (Last 3 turns) ==="
            for action in self.recent_actions[-3:]:
                formatted += f"\nTurn {action['turn']}: {action['actions']} → {action['result']}"
            formatted += "\n=== END RECENT ACTIONS ==="

        # Add context information
        context = extract_game_context(obs)
        if context["in_combat"]:
            formatted += "\n\n⚔️ COMBAT ALERT: Monster nearby!"
        if context["low_health"]:
            formatted += "\n\n❤️ WARNING: Low health!"
        if context["hungry"]:
            formatted += "\n\n🍖 WARNING: You are hungry!"
        if context["at_stairs"]:
            formatted += f"\n\n🪜 You are at the {context.get('stairs_type', 'stairs')}!"

        # Add ALL valid actions organized by category
        from src.synth_env.examples.nethack.helpers.action_mapping import (
            ACTION_CATEGORIES,
        )

        formatted += "\n\n=== VALID ACTIONS ==="
        for category in ACTION_CATEGORIES:
            formatted += f"\n{category.name} ({category.description}):"
            formatted += f"\n  {', '.join(category.actions)}"

        # Also add context-specific suggestions
        suggested_actions = get_actions_for_context(obs)
        if suggested_actions:
            formatted += f"\n\n💡 Suggested for current situation: {', '.join(suggested_actions)}"

        return formatted

    async def decide(self, obs: str) -> Dict[str, Any]:
        """Get LLM decision based on observation."""
        try:
            # Debug: log first few observations
            if len(self.history) < 3:
                logger.info(f"Turn {len(self.history) + 1} observation preview: {obs[:300]}...")

            # Save full prompt to file for inspection
            turn_num = len(self.history) + 1
            with open(f"nethack_prompt_turn_{turn_num}.txt", "w") as f:
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(self.system_prompt)
                f.write("\n\n=== USER MESSAGE (OBSERVATION) ===\n")
                f.write(obs)
                f.write("\n\n=== TOOLS ===\n")
                f.write(json.dumps([tool.to_openai_tool() for tool in self.tools], indent=2))

            # Add observation to history (limit history size)
            self.history.append({"role": "user", "content": obs})
            if len(self.history) > 10:
                # Keep only recent history
                self.history = self.history[-10:]

            # Get LLM response
            response = await self.llm.respond_async(
                system_message=self.system_prompt, user_message=obs, tools=self.tools
            )

            # Check response has tool calls
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, "tool_calls"):
                logger.info(f"tool_calls value: {response.tool_calls}")

            # Parse response - access tool_calls directly like other agents
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                logger.info(f"Found {len(tool_calls)} tool calls")

                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]

                    # Handle different tool call structures
                    tool_name = ""
                    tool_args_str = ""

                    if (
                        hasattr(tool_call, "function")
                        and hasattr(tool_call.function, "name")
                        and hasattr(tool_call.function, "arguments")
                    ):
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                    elif (
                        isinstance(tool_call, dict)
                        and "function" in tool_call
                        and isinstance(tool_call["function"], dict)
                    ):
                        tool_name = tool_call["function"].get("name")
                        tool_args_str = tool_call["function"].get("arguments")

                    # Log the full tool call for debugging
                    logger.info(f"Tool name: {tool_name}, Args: {tool_args_str}")

                    # Parse arguments
                    if isinstance(tool_args_str, str):
                        try:
                            args = json.loads(tool_args_str)
                        except:
                            args = {
                                "reasoning": "Failed to parse arguments",
                                "actions": ["wait"],
                            }
                    else:
                        args = tool_args_str

                    # Track this action for next turn's context
                    turn_num = len(self.recent_actions) + 1
                    action_record = {
                        "turn": turn_num,
                        "actions": args.get("actions", [args.get("action", "unknown")]),
                        "reasoning": args.get("reasoning", ""),
                        "result": "pending",  # Will be updated after execution
                    }
                    self.recent_actions.append(action_record)

                    # Keep only last 3 actions
                    if len(self.recent_actions) > 3:
                        self.recent_actions = self.recent_actions[-3:]

                    return {"name": tool_name, "parameters": args}

            # Fallback to exploring
            logger.warning("No tool call found in LLM response, defaulting to wait")
            logger.warning(f"Response type: {type(response)}")
            logger.warning(f"Response attrs: {dir(response)}")
            logger.warning(f"Response: {response}")

            # Log all attributes of response for debugging
            for attr in dir(response):
                if not attr.startswith("_"):
                    try:
                        value = getattr(response, attr)
                        if not callable(value):
                            logger.warning(f"  {attr}: {value}")
                    except:
                        pass
            return {
                "name": "nethack_interact",
                "parameters": {
                    "reasoning": "No valid response from LLM",
                    "actions": ["wait"],
                },
            }

        except Exception as e:
            logger.error(f"Error in decide: {e}")
            # Default safe action
            return {
                "name": "nethack_interact",
                "parameters": {"reasoning": f"Error: {str(e)}", "actions": ["wait"]},
            }

    async def run_episode(self, env: NetHackEnvironment) -> Dict[str, Any]:
        """Run one episode with the agent."""
        # Get task instructions
        self.system_prompt = self._create_system_prompt(env.task_instance.impetus.instructions)

        # Initialize environment
        obs = await env.initialize()

        # Track episode statistics
        stats = {
            "turns": 0,
            "max_depth": 1,
            "final_score": 0,
            "total_reward": 0.0,
            "balrog_total_reward": 0.0,
            "balrog_score": 0.0,
            "terminated": False,
            "death_reason": None,
            "objectives_completed": 0,
            "achievements_unlocked": [],
            "achievement_details": {},
            "error": None,
            "actions_taken": [],  # Track all actions
            "observations": [],  # Track key observations
        }

        try:
            for turn in range(self.max_turns):
                stats["turns"] = turn + 1

                # Format observation for agent
                formatted_obs = self._format_observation(obs)

                # Get agent decision
                action = await self.decide(formatted_obs)

                # Record the action
                action_record = {
                    "turn": turn + 1,
                    "action_type": action["name"],
                    "action": "unknown",  # Will be filled in later
                    "action_params": action.get("parameters", {}),
                    "position_before": obs.get("position", "unknown"),
                    "dungeon_level": obs.get("dungeon_level", 1),
                }

                # Check for termination
                if action["name"] == "terminate":
                    stats["terminated"] = True
                    stats["death_reason"] = action["parameters"].get("reason", "Agent terminated")
                    action_record["action"] = "terminate"
                    action_record["result"] = "terminated"
                    stats["actions_taken"].append(action_record)
                    break

                # Execute action(s)
                if action["name"] == "nethack_interact":
                    params = action["parameters"]

                    # Log reasoning
                    reasoning = params.get("reasoning", "No reasoning provided")
                    logger.info(f"Reasoning: {reasoning}")

                    # Handle both old format (single action) and new format (multiple actions)
                    if "actions" in params:
                        actions_list = params["actions"]
                    elif "action" in params:
                        actions_list = [params["action"]]
                    else:
                        actions_list = ["wait"]

                    # Execute each action in sequence
                    for act in actions_list:
                        if obs.get("terminated", False):
                            break

                        # Handle "fight" by converting to movement
                        if act == "fight":
                            logger.warning(
                                "'fight' is not a valid action - to attack, move into the monster!"
                            )
                            # Skip this action
                            continue

                        # Save position before this specific action
                        pos_before = obs.get("position", "unknown")

                        obs = await env.step(act)

                        # Update the recent action result with the message from this action
                        result_msg = obs.get("message", "").rstrip("\x00").strip()
                        if not result_msg:
                            result_msg = "No message"
                        self._update_last_action_result(result_msg)

                        # Create a new record for each action
                        single_action_record = {
                            "turn": stats["turns"],
                            "action_type": "nethack_interact",
                            "action": act,
                            "reasoning": reasoning if act == actions_list[0] else "continuation",
                            "position_before": pos_before,
                            "position_after": obs.get("position", "unknown"),
                            "message": obs.get("message", "").rstrip("\x00")[:100],
                            "reward": obs.get("reward_last", 0),
                            "hp": obs.get("character_stats", {}).get("hp", "unknown"),
                        }
                        stats["actions_taken"].append(single_action_record)
                else:
                    logger.warning(f"Unknown action: {action['name']}")
                    obs = await env.step("wait")
                    action_record["action"] = "wait (fallback)"
                    stats["actions_taken"].append(action_record)

                # Update statistics - expect these fields to exist
                stats["max_depth"] = max(stats["max_depth"], obs["dungeon_level"])
                stats["total_reward"] = obs["total_reward"]
                stats["final_score"] = obs["score"]

                # Track achievements and Balrog rewards
                if "achievements_unlocked" in obs:
                    for ach, unlocked in obs["achievements_unlocked"].items():
                        if unlocked and ach not in stats["achievements_unlocked"]:
                            stats["achievements_unlocked"].append(ach)

                if "balrog_total_reward" in obs:
                    stats["balrog_total_reward"] = obs["balrog_total_reward"]

                if "achievement_stats" in obs and "balrog_score" in obs["achievement_stats"]:
                    stats["balrog_score"] = obs["achievement_stats"]["balrog_score"]

                # Update the last observation for next iteration
                # (removed duplicate action record append)

                # Record key observations every 5 turns
                if turn % 5 == 0 or turn == 0:
                    stats["observations"].append(
                        {
                            "turn": turn + 1,
                            "position": obs.get("position", "unknown"),
                            "dungeon_level": obs.get("dungeon_level", 1),
                            "hp": f"{obs.get('character_stats', {}).get('hp', '?')}/{obs.get('character_stats', {}).get('max_hp', '?')}",
                            "score": obs.get("score", 0),
                            "message": obs.get("message", "")[:100],
                        }
                    )

                # Check for game termination
                if obs["terminated"]:
                    stats["terminated"] = True
                    if "died" in obs["message"].lower():
                        stats["death_reason"] = "Character died"
                    else:
                        stats["death_reason"] = "Game ended"
                    break

                # Check if objective achieved
                # We know metadata is NetHackTaskInstanceMetadata from environment
                metadata = cast(NetHackTaskInstanceMetadata, env.task_instance.metadata)
                target_depth = metadata.target_depth
                if obs["dungeon_level"] >= target_depth:
                    logger.info(f"Objective achieved! Reached depth {target_depth}")
                    stats["objectives_completed"] += 1

        except Exception as e:
            logger.error(f"Error during episode: {e}")
            stats["error"] = str(e)

        finally:
            # Ensure environment is terminated
            await env.terminate()

        return stats


async def eval_react_nethack(
    model_name: str = "gpt-4.1-nano", num_episodes: int = 3, max_turns: int = 50
) -> List[Dict[str, Any]]:
    """Run ReAct agent evaluation on NetHack taskset."""
    logger.info(f"Starting NetHack evaluation with model: {model_name}")

    # Load taskset
    taskset = await create_nethack_taskset()
    logger.info(f"Loaded {len(taskset.instances)} task instances")

    # Initialize LLM and agent
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.7)
    agent = NetHackReActAgent(llm, max_turns=max_turns)

    # Select subset of tasks for evaluation
    # Focus on tasks that require actual exploration (target_depth > 1)
    eval_instances = [
        inst
        for inst in taskset.instances
        if hasattr(inst.metadata, "difficulty")
        and hasattr(inst.metadata, "target_depth")
        and cast(NetHackTaskInstanceMetadata, inst.metadata).target_depth > 1
        and cast(NetHackTaskInstanceMetadata, inst.metadata).difficulty
        in ["beginner", "intermediate"]
    ][:num_episodes]

    logger.info(f"Evaluating on {len(eval_instances)} instances")

    # Run episodes
    results = []
    for i, instance in enumerate(eval_instances):
        logger.info(f"\nEpisode {i + 1}/{len(eval_instances)}")
        if hasattr(instance.metadata, "character_role"):
            metadata = cast(NetHackTaskInstanceMetadata, instance.metadata)
            logger.info(f"Character: {metadata.character_role}")
            logger.info(f"Target depth: {metadata.target_depth}")
            logger.info(f"Time limit: {metadata.time_limit} turns")

        try:
            env = NetHackEnvironment(instance)
            result = await agent.run_episode(env)

            # Add task info to result
            result["task_id"] = str(instance.id)
            if hasattr(instance.metadata, "character_role"):
                metadata = cast(NetHackTaskInstanceMetadata, instance.metadata)
                result["character_role"] = metadata.character_role
                result["target_depth"] = metadata.target_depth
                result["time_limit"] = metadata.time_limit
                result["difficulty"] = metadata.difficulty
                result["success"] = result["max_depth"] >= metadata.target_depth
            else:
                result["success"] = False

            results.append(result)

            logger.info(
                f"Episode completed - Success: {result['success']}, "
                f"Depth: {result['max_depth']}/{result['target_depth']}, "
                f"Turns: {result['turns']}, Score: {result['final_score']}, "
                f"Balrog: {result.get('balrog_score', 0):.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to run episode: {e}")
            results.append({"task_id": str(instance.id), "error": str(e), "success": False})

    return results


def analyze_nethack_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze NetHack-specific performance metrics."""
    if not results:
        return {}

    # Filter out errored results for metrics
    valid_results = [r for r in results if "error" not in r or r["error"] is None]

    if not valid_results:
        return {"error_rate": 1.0, "num_episodes": len(results)}

    metrics = {
        "num_episodes": len(results),
        "success_rate": sum(1 for r in valid_results if r.get("success", False))
        / len(valid_results),
        "avg_depth_reached": sum(r.get("max_depth", 1) for r in valid_results) / len(valid_results),
        "avg_turns": sum(r.get("turns", 0) for r in valid_results) / len(valid_results),
        "avg_score": sum(r.get("final_score", 0) for r in valid_results) / len(valid_results),
        "avg_reward": sum(r.get("total_reward", 0.0) for r in valid_results) / len(valid_results),
        "avg_balrog_reward": sum(r.get("balrog_total_reward", 0.0) for r in valid_results)
        / len(valid_results),
        "avg_balrog_score": sum(r.get("balrog_score", 0.0) for r in valid_results)
        / len(valid_results),
        "death_rate": sum(
            1 for r in valid_results if "died" in str(r.get("death_reason", "")).lower()
        )
        / len(valid_results),
        "timeout_rate": sum(
            1 for r in valid_results if r.get("turns", 0) >= r.get("time_limit", float("inf"))
        )
        / len(valid_results),
        "error_rate": sum(1 for r in results if "error" in r and r["error"] is not None)
        / len(results),
        # Achievement metrics
        "avg_achievements_unlocked": sum(
            len(r.get("achievements_unlocked", [])) for r in valid_results
        )
        / len(valid_results),
        "total_unique_achievements": len(
            set(ach for r in valid_results for ach in r.get("achievements_unlocked", []))
        ),
    }

    # Count how many times each achievement was unlocked
    achievement_counts = {}
    for r in valid_results:
        for ach in r.get("achievements_unlocked", []):
            achievement_counts[ach] = achievement_counts.get(ach, 0) + 1

    # Most common achievements
    if achievement_counts:
        sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)
        metrics["most_common_achievements"] = sorted_achievements[:5]

    # Breakdown by difficulty
    for difficulty in ["tutorial", "beginner", "intermediate", "advanced", "expert"]:
        diff_results = [r for r in valid_results if r.get("difficulty") == difficulty]
        if diff_results:
            metrics[f"{difficulty}_success_rate"] = sum(
                1 for r in diff_results if r.get("success", False)
            ) / len(diff_results)
            metrics[f"{difficulty}_avg_depth"] = sum(
                r.get("max_depth", 1) for r in diff_results
            ) / len(diff_results)
            metrics[f"{difficulty}_avg_achievements"] = sum(
                len(r.get("achievements_unlocked", [])) for r in diff_results
            ) / len(diff_results)

    # Breakdown by character role
    for role in ["tourist", "knight", "wizard", "barbarian"]:
        role_results = [r for r in valid_results if r.get("character_role") == role]
        if role_results:
            metrics[f"{role}_success_rate"] = sum(
                1 for r in role_results if r.get("success", False)
            ) / len(role_results)
            metrics[f"{role}_avg_achievements"] = sum(
                len(r.get("achievements_unlocked", [])) for r in role_results
            ) / len(role_results)

    return metrics


async def main():
    """Run the evaluation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate NetHack ReAct agent")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="Model to use (default: gpt-4.1-nano)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum turns per episode (default: 50)",
    )

    args = parser.parse_args()

    print(f"Running NetHack evaluation with:")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max turns: {args.max_turns}")

    # Run evaluation
    results = await eval_react_nethack(
        model_name=args.model, num_episodes=args.episodes, max_turns=args.max_turns
    )

    # Analyze results
    metrics = analyze_nethack_results(results)

    # Print results - OFFICIAL LEADERBOARD SCORE FIRST
    print("\n" + "=" * 80)
    print("🏆 OFFICIAL BALROG LEADERBOARD SCORE 🏆")
    print("=" * 80)
    print(f"📊 BALROG SCORE (0-100%): {metrics.get('avg_balrog_score', 0):.3f}%")
    print(f"📈 Current SOTA benchmark: ~1-2%")
    print(f"🎯 Episodes evaluated: {metrics.get('num_episodes', 0)}")
    print("=" * 80)
    print("⚠️  This is the ONLY score that matters for SOTA claims and leaderboard comparisons!")
    print("⚠️  All other metrics below are for analysis/debugging only.")

    print("\n" + "=" * 80)
    print("📋 ANALYSIS METRICS (Not for leaderboard comparison)")
    print("=" * 80)
    print(f"Success rate (task completion): {metrics.get('success_rate', 0):.2%}")
    print(f"Average depth reached: {metrics.get('avg_depth_reached', 0):.1f}")
    print(f"Average turns: {metrics.get('avg_turns', 0):.0f}")
    print(f"Average game score: {metrics.get('avg_score', 0):.0f}")
    print(f"Death rate: {metrics.get('death_rate', 0):.2%}")
    print(f"Error rate: {metrics.get('error_rate', 0):.2%}")

    print("\n=== Training Signal Metrics (Shaped Rewards) ===")
    print(f"Average custom reward: {metrics.get('avg_reward', 0):.2f}")
    print(f"Average Balrog shaped reward: {metrics.get('avg_balrog_reward', 0):.2f}")
    print("(These are training signals, NOT the leaderboard score)")

    print("\n=== Achievement Metrics ===")
    print(f"Average achievements unlocked: {metrics.get('avg_achievements_unlocked', 0):.1f}")
    print(f"Total unique achievements: {metrics.get('total_unique_achievements', 0)}")

    if "most_common_achievements" in metrics and metrics["most_common_achievements"]:
        print("\nMost common achievements:")
        for ach, count in metrics["most_common_achievements"]:
            print(f"  {ach}: {count} times ({count / metrics['num_episodes'] * 100:.0f}%)")

    # Save results
    with open("nethack_react_results.json", "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)

    print("\nResults saved to nethack_react_results.json")

    # Print detailed action summary for sanity check
    print("\n=== Detailed Episode Summary ===")
    for i, result in enumerate(results):
        print(f"\nEpisode {i + 1}:")
        print(f"  🏆 BALROG LEADERBOARD SCORE: {result.get('balrog_score', 0):.3f}%")
        print(f"  Character: {result.get('character_role', 'unknown')}")
        print(f"  Target depth: {result.get('target_depth', 'unknown')}")
        print(f"  Max depth reached: {result.get('max_depth', 0)}")
        print(f"  Total turns: {result.get('turns', 0)}")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Final game score: {result.get('final_score', 0)}")
        print(f"  Custom shaped reward: {result.get('total_reward', 0):.2f}")
        print(f"  Balrog shaped reward: {result.get('balrog_total_reward', 0):.2f}")

        if result.get("achievements_unlocked"):
            print(f"  Achievements unlocked ({len(result['achievements_unlocked'])}):")
            for ach in result["achievements_unlocked"][:10]:  # Show first 10
                print(f"    - {ach}")
            if len(result["achievements_unlocked"]) > 10:
                print(f"    ... and {len(result['achievements_unlocked']) - 10} more")

        if "actions_taken" in result and result["actions_taken"]:
            print(f"\n  First 10 actions:")
            for action in result["actions_taken"][:10]:
                print(
                    f"    Turn {action.get('turn', '?')}: {action.get('action', 'unknown')} "
                    f"(pos: {action.get('position_before', '?')} → {action.get('position_after', '?')}, "
                    f"HP: {action.get('hp', '?')})"
                )
                if action.get("reasoning") and action["reasoning"] != "continuation":
                    print(f"      Reasoning: {action['reasoning'][:80]}...")
                if action.get("message", "").strip():
                    print(f"      Message: {action['message'][:60]}...")

        if "observations" in result and result["observations"]:
            print(f"\n  Key observations:")
            for obs in result["observations"]:
                print(
                    f"    Turn {obs['turn']}: Level {obs['dungeon_level']}, "
                    f"HP: {obs['hp']}, Score: {obs['score']}"
                )


if __name__ == "__main__":
    asyncio.run(main())
