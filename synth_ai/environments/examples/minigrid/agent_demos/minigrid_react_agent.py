"""ReAct agent demo for MiniGrid environment."""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import uuid

# Import SynthAI LM and BaseTool
from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_sdk.tracing.decorators import trace_event_async
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth_ai.environments.examples.minigrid.environment import MiniGridEnvironment
from synth_ai.environments.examples.minigrid.taskset import (
    create_minigrid_taskset,
    DEFAULT_MINIGRID_TASK,
)
from synth_ai.environments.environment.tools import EnvToolCall


# --- Pydantic Models for Tool Arguments ---
class MiniGridActArgs(BaseModel):
    """Arguments for MiniGrid action."""

    action: str = Field(
        description="The action to take. Must be one of: 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'"
    )
    reasoning: str = Field(description="A brief explanation of why this action was chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


# --- Tool Definitions ---


class MiniGridActTool(BaseTool):
    """Tool for performing an action in MiniGrid."""

    name: str = "minigrid_act"
    arguments: type[BaseModel] = MiniGridActArgs
    description: str = "Perform an action in the MiniGrid environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- ReAct Agent ---
class MiniGridReActAgent:
    """ReAct agent for MiniGrid environments."""

    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.debug_log = []  # Store all prompts and responses for debugging
        self.system_name: str = "minigrid-react-agent"  # Required for synth-sdk tracing
        self.system_instance_id: str = str(uuid.uuid4())  # Required for synth-sdk tracing

        # Available tools
        self.tools = [MiniGridActTool(), TerminateTool()]

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM."""
        if "observation" in obs:
            return obs["observation"]

        # Fallback formatting
        parts = []
        if "mission" in obs:
            parts.append(f"Mission: {obs['mission']}")
        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")
        if "reward_last" in obs:
            parts.append(f"Last Reward: {obs['reward_last']:.3f}")
        if "total_reward" in obs:
            parts.append(f"Total Reward: {obs['total_reward']:.3f}")

        return "\n".join(parts)

    @trace_event_async(event_type="minigrid_react_decide")
    async def decide(self, obs: str, task_description: str, turn: int) -> Dict[str, Any]:
        """Get LLM decision for next action."""
        system_message = f"""You are playing a MiniGrid environment. {task_description}

CRITICAL UNDERSTANDING OF THE GRID:

1. HOW TO READ THE GRID:
   - The grid shows a top-down view of a small world
   - Your position is shown by an arrow: → ↓ ← ↑
   - The arrow shows both WHERE you are and WHICH DIRECTION you're facing

2. GRID SYMBOLS:
   - → ↓ ← ↑ = YOU (the arrow points in the direction you're facing)
   - # = wall (CANNOT move through these)
   - . = empty space (CAN move through these)
   - G = goal (your target - GET HERE to win!)
   - L = lava (AVOID - stepping on this ends the game)
   - K = key, D = door, B = ball (for special levels)
   - ? = edge of the grid (CANNOT move here - it's the boundary)

3. HOW MOVEMENT WORKS:
   - 'forward' = move ONE space in the direction your arrow is pointing
   - 'left' = turn 90 degrees left (changes arrow direction, doesn't move you)
   - 'right' = turn 90 degrees right (changes arrow direction, doesn't move you)
   - You CANNOT move through walls (#) or boundaries (?)
   
4. DEBUG MESSAGES:
   - "Forward blocked by wall" = you tried to move into a wall
   - "Forward blocked by boundary" = you tried to move outside the grid
   - "Moved forward" = you successfully moved

5. IMPORTANT - LIMITED VISIBILITY:
   - You have LIMITED VISION and can only see a small area around you
   - The goal (G) might NOT be visible initially - you need to EXPLORE
   - The ? symbols show areas beyond your current view
   - You must move around the maze to discover new areas

6. EXPLORATION STRATEGY:
   - If you DON'T see the goal (G), you must EXPLORE the maze
   - Move systematically through empty spaces (.) to reveal new areas
   - Try to explore unexplored paths rather than revisiting the same spots
   - Keep track of where you've been to avoid going in circles
   - When you discover the goal (G), then plan a path to reach it

7. LEARN FROM PAST ACTIONS:
   - If an action was blocked, DON'T repeat it immediately
   - If you keep getting blocked moving forward, try turning left or right
   - If you're stuck in a pattern, break it by trying a different approach"""

        # Extract debug information to highlight it
        debug_info = ""
        if "Debug:" in obs:
            debug_lines = [
                line
                for line in obs.split("\n")
                if "Debug:" in line or "Last action result:" in line
            ]
            if debug_lines:
                debug_info = (
                    "\n\n🚨 IMPORTANT DEBUG INFORMATION:\n"
                    + "\n".join(f"• {line}" for line in debug_lines)
                    + "\n"
                )

        # Build action history string
        action_history = ""
        if len(self.history) > 0:
            action_history = "\n\nRECENT HISTORY (Last 3 Actions):\n"
            for i, h in enumerate(self.history[-3:], 1):
                action_history += f"{i}. {h}\n"
            action_history += "\nBased on this history, avoid repeating failed actions and learn from what worked!\n"

        user_content = f"Current state:\n{obs}{debug_info}{action_history}\nCRITICAL: Check the debug information above! If blocked by wall, you MUST turn or try a different action.\n\nWhat action should I take?"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        # Log the prompt
        prompt_entry = {
            "turn": turn,
            "type": "prompt",
            "messages": messages,
            "tools": self.tools,
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(prompt_entry)

        response = await self.llm.respond_async(
            messages=messages,
            tools=self.tools,
        )

        # Log the response
        response_entry = {
            "turn": turn,
            "type": "llm_response",
            "response": str(response),
            "response_type": type(response).__name__,
            "tool_calls": getattr(response, "tool_calls", None),
            "content": getattr(response, "content", None),
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(response_entry)

        # Debug: Print response type
        if self.verbose:
            print(f"DEBUG: LLM response type: {type(response)}")
            print(f"DEBUG: LLM response full: {response}")
            if hasattr(response, "tool_calls"):
                print(f"DEBUG: Tool calls: {response.tool_calls}")
                if response.tool_calls:
                    print(f"DEBUG: First tool call: {response.tool_calls[0]}")
                    print(f"DEBUG: First tool call type: {type(response.tool_calls[0])}")
            if hasattr(response, "content"):
                print(f"DEBUG: Response content: {response.content}")

        # Parse tool calls - fail fast, no defensive programming
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            # Handle different response formats
            if isinstance(tool_call, dict):
                # Dict format from LLM
                func = tool_call["function"]
                action = {
                    "name": func["name"],
                    "parameters": json.loads(func["arguments"]),
                }
            elif hasattr(tool_call, "function"):
                # Object format
                action = {
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                }
            else:
                # Unexpected format - fail fast
                raise ValueError(f"Unexpected tool_call format: {tool_call}")
        else:
            # No tool call - fail fast
            raise ValueError("No tool call returned from LLM")

        # Log the parsed action
        action_entry = {
            "turn": turn,
            "type": "parsed_action",
            "action": action,
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(action_entry)

        return action

    @trace_event_async(event_type="minigrid_react_episode")
    async def run_episode(self, env: MiniGridEnvironment) -> Dict[str, Any]:
        """Run one episode in the environment."""
        # Initialize
        obs = await env.initialize()
        task_description = env.task_instance.impetus.instructions

        if self.verbose:
            print(f"\nTask: {task_description}")
            print(f"Initial observation:\n{self._format_observation(obs)}\n")

        success = False
        total_reward = 0.0
        last_reward = 0.0

        for turn in range(self.max_turns):
            # Format observation
            formatted_obs = self._format_observation(obs)

            # Log the observation
            obs_entry = {
                "turn": turn,
                "type": "observation",
                "raw_obs": obs,
                "formatted_obs": formatted_obs,
                "timestamp": datetime.now().isoformat(),
            }
            self.debug_log.append(obs_entry)

            # Get agent decision
            action = await self.decide(formatted_obs, task_description, turn)

            if self.verbose:
                print(f"\nTurn {turn + 1}:")
                print(f"Action: {action['name']}")
                if "parameters" in action:
                    print(f"Parameters: {action['parameters']}")

            # Check for termination
            if action["name"] == "terminate":
                if self.verbose:
                    print(f"Agent terminated: {action['parameters']['reason']}")
                break

            # Execute action
            tool_call = {"tool": action["name"], "args": action["parameters"]}

            # Log the tool call
            tool_call_entry = {
                "turn": turn,
                "type": "tool_call",
                "tool_call": tool_call,
                "timestamp": datetime.now().isoformat(),
            }
            self.debug_log.append(tool_call_entry)

            # Debug: Print tool call
            if self.verbose:
                print(f"DEBUG: Sending tool_call: {tool_call}")

            obs = await env.step(tool_call)

            # Log the environment response
            env_response_entry = {
                "turn": turn,
                "type": "env_response",
                "response": obs,
                "timestamp": datetime.now().isoformat(),
            }
            self.debug_log.append(env_response_entry)

            # Debug: Print response
            if self.verbose:
                print(f"DEBUG: Environment response keys: {list(obs.keys())}")
                if "error" in obs:
                    print(f"DEBUG: ERROR: {obs['error']}")

            # Track history with result
            action_taken = action["parameters"]["action"]
            action_reasoning = action["parameters"]["reasoning"]
            action_result = obs["last_action_result"]

            # Extract position info if available
            position_info = ""
            if "observation" in obs:
                lines = obs["observation"].split("\n")
                for line in lines:
                    if "Agent Position" in line:
                        position_info = f" -> {line}"
                        break

            history_entry = f"Action: {action_taken} | Reasoning: {action_reasoning} | Result: {action_result}{position_info}"
            self.history.append(history_entry)

            # Update metrics
            total_reward = obs["total_reward"]
            last_reward = obs["reward_last"]

            if self.verbose:
                print(f"Reward: {last_reward:.3f} (Total: {total_reward:.3f})")
                if "observation" in obs:
                    # Just print position line for brevity
                    lines = obs["observation"].split("\n")
                    for line in lines:
                        if "Agent Position" in line:
                            print(line)
                            break

            # Check if terminated
            if obs["terminated"]:
                success = obs["success"] or "goal" in str(obs).lower()
                if self.verbose:
                    print(f"\nEpisode ended! Success: {success}, Final Reward: {total_reward:.3f}")
                break

        # Get final metrics
        final_obs = await env.terminate()

        # Log final episode summary
        episode_summary = {
            "type": "episode_summary",
            "success": success,
            "turns": turn + 1,
            "total_reward": total_reward,
            "final_position": final_obs["final_position"],
            "total_steps": final_obs["total_steps"],
            "debug_log_entries": len(self.debug_log),
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(episode_summary)

        return {
            "success": success,
            "turns": turn + 1,
            "total_reward": total_reward,
            "final_position": final_obs["final_position"],
            "total_steps": final_obs["total_steps"],
            "debug_log": self.debug_log,  # Include full debug log
        }


# --- Evaluation Function ---
@trace_event_async(event_type="eval_minigrid_react")
async def eval_minigrid_react(
    model_name: str = "gpt-4-mini",
    num_tasks: int = 5,
    difficulty: str = "easy",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate ReAct agent on MiniGrid tasks."""
    # Generate task set
    taskset = await create_minigrid_taskset(
        num_tasks_per_difficulty={difficulty: num_tasks}, seed=42
    )

    # Initialize LLM and agent
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.7)
    agent = MiniGridReActAgent(llm, max_turns=15, verbose=verbose)  # Reduced max turns

    # Create debug logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"minigrid_debug_logs_{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)

    # Run evaluation
    results = []
    all_debug_logs = []

    for i, task in enumerate(taskset.instances[:num_tasks]):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Task {i + 1}/{num_tasks}: {task.metadata.env_name}")
            print(f"{'=' * 60}")

        # Create environment
        env = MiniGridEnvironment(task)

        # Run episode
        result = await agent.run_episode(env)
        result["task_id"] = str(task.id)
        result["env_name"] = task.metadata.env_name
        result["difficulty"] = task.metadata.difficulty

        # Save debug log for this task
        debug_log = result.pop("debug_log", [])  # Remove from result to avoid duplication
        debug_log_file = os.path.join(
            debug_dir, f"task_{i + 1}_{model_name.replace('.', '_')}_debug.json"
        )
        with open(debug_log_file, "w") as f:
            json.dump(
                {
                    "task_info": {
                        "task_id": result["task_id"],
                        "env_name": result["env_name"],
                        "difficulty": result["difficulty"],
                        "model": model_name,
                    },
                    "result": result,
                    "debug_log": debug_log,
                },
                f,
                indent=2,
                default=str,
            )

        all_debug_logs.append(debug_log)
        results.append(result)

        if verbose:
            print(f"\nResult: {result}")
            print(f"Debug log saved to: {debug_log_file}")

    # Save summary debug info
    summary_debug_file = os.path.join(debug_dir, f"summary_{model_name.replace('.', '_')}.json")
    with open(summary_debug_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "timestamp": timestamp,
                "all_debug_logs": all_debug_logs,
            },
            f,
            indent=2,
            default=str,
        )

    # Compute statistics
    successes = [r["success"] for r in results]
    success_rate = sum(successes) / len(successes) if successes else 0
    avg_reward = sum(r["total_reward"] for r in results) / len(results) if results else 0
    avg_steps = sum(r["total_steps"] for r in results) / len(results) if results else 0

    summary = {
        "model": model_name,
        "num_tasks": len(results),
        "difficulty": difficulty,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "results": results,
        "debug_dir": debug_dir,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Steps: {avg_steps:.1f}")

    return summary


# --- Main ---
async def main():
    """Run the demo."""
    print("Testing MiniGrid ReAct Agent")
    print("=" * 60)

    # Models to test
    models = ["gpt-4.1-nano", "gpt-4.1-mini"]
    all_results = {}

    for model in models:
        print(f"\n\n{'=' * 60}")
        print(f"Testing model: {model}")
        print(f"{'=' * 60}")

        # Run evaluation
        summary = await eval_minigrid_react(
            model_name=model,
            num_tasks=5,  # Run 5 tasks per model
            difficulty="easy",
            verbose=True,
        )

        all_results[model] = summary

        # Print model summary
        print(f"\n\nSummary for {model}:")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(f"Average Steps: {summary['avg_steps']:.1f}")

    # Compare results
    print(f"\n\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Success Rate':<15} {'Avg Reward':<15} {'Avg Steps':<10}")
    print("-" * 60)
    for model, summary in all_results.items():
        print(
            f"{model:<20} {summary['success_rate']:.1%}{'':10} "
            f"{summary['avg_reward']:.3f}{'':10} "
            f"{summary['avg_steps']:.1f}"
        )

    # Detailed results
    print("\n\nDetailed Results:")
    for model, summary in all_results.items():
        print(f"\n{model}:")
        for i, result in enumerate(summary["results"]):
            print(
                f"  Task {i + 1}: Success={result['success']}, "
                f"Reward={result['total_reward']:.3f}, "
                f"Steps={result['total_steps']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
