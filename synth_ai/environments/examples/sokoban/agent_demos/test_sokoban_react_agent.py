#!/usr/bin/env python3
"""
Test script to run ReAct agents against Sokoban environment on synth service (port 8901)
Tests gemini-1.5-flash on multiple easy Sokoban instances
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

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool


# --- Service Configuration ---
SERVICE_BASE_URL = "http://localhost:8901"
MODEL_NAME = "gpt-4.1-mini"
NUM_INSTANCES = 10
MAX_TURNS = 15
DIFFICULTY = "ultra_easy"

# ultra easy - gpt-4.1-nano - 0%, gpt-4.1-mini - 16%, o4-mini - 84%
# easy - o4-mini - 10%

# --- Action Mapping ---
ACTION_STRING_TO_INT = {
    "no operation": 0,
    "push up": 1,
    "push down": 2,
    "push left": 3,
    "push right": 4,
    "move up": 5,
    "move down": 6,
    "move left": 7,
    "move right": 8,
}


# --- Tool Definitions ---
class GameActionArgs(BaseModel):
    """Arguments for game actions."""

    action: str = Field(description="The action to take")
    reasoning: str = Field(description="Brief explanation of why this action was chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


class GameActionTool(BaseTool):
    """Tool for performing an action in the game."""

    name: str = "game_action"
    arguments: type[BaseModel] = GameActionArgs
    description: str = "Perform an action in the game environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for game environments."""

    def __init__(self, llm: LM, max_turns: int = MAX_TURNS, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"
        self.system_instance_id = str(uuid.uuid4())
        self.tools = [GameActionTool(), TerminateTool()]

    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get LLM decision for next action."""
        # Build action history (only last 2 for brevity)
        action_history = ""
        if len(self.history) > 0:
            action_history = "\n\nRECENT HISTORY:\n"
            for i, h in enumerate(self.history[-2:], 1):
                action_history += f"{i}. {h}\n"

        user_content = f"Current state:\n{obs}{action_history}\n\nWhat action should I take?"

        # Use the same pattern as Crafter ReAct agent
        response_obj = await self.llm.respond_async(
            system_message=system_message, user_message=user_content, tools=self.tools
        )

        tool_calls = response_obj.tool_calls

        # Handle case where tool_calls is None or empty (graceful fallback)
        if not tool_calls:
            if self.verbose:
                print(f"[WARNING] No tool calls returned by LLM, using default action")
            return {
                "name": "game_action",
                "parameters": {
                    "action": "up",
                    "reasoning": "Default action - no tool call received",
                },
            }

        tool_call_data = tool_calls[0]

        # Handle both dict and object formats (same as Crafter)
        if isinstance(tool_call_data, dict):
            tool_name = tool_call_data["function"]["name"]
            tool_args_str = tool_call_data["function"]["arguments"]
        else:
            tool_name = tool_call_data.function.name
            tool_args_str = tool_call_data.function.arguments

        tool_arguments = json.loads(tool_args_str)

        return {"name": tool_name, "parameters": tool_arguments}


# --- Sokoban ReAct Agent ---
class SokobanReActAgent(BaseReActAgent):
    """ReAct agent for Sokoban environment."""

    def __init__(self, llm: LM, max_turns: int = 15, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "sokoban-react-agent"

    def get_system_message(self) -> str:
        return """You are playing Sokoban. Push all boxes (X) onto targets (O) to win.

RULES: Move/push in 4 directions. Cannot pull boxes or push into walls/boxes.

ACTIONS: "move up", "move down", "move left", "move right", "push up", "push down", "push left", "push right", "no operation"

SYMBOLS: # = wall, _ = empty, O = target, X = box, √ = box on target, P = you

STRATEGY: Analyze layout, plan moves, avoid getting boxes stuck in corners. Use PUSH actions when next to a box to move it.

Be concise and decisive. Always use the exact action names listed above."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for Sokoban."""
        parts = []

        if "room_text" in obs:
            parts.append(f"Board:\n{obs['room_text']}")

        if "boxes_on_target" in obs and "num_boxes" in obs:
            parts.append(f"Progress: {obs['boxes_on_target']}/{obs['num_boxes']} boxes on target")

        if "steps_taken" in obs and "max_steps" in obs:
            parts.append(f"Steps: {obs['steps_taken']}/{obs['max_steps']}")

        return "\n".join(parts)


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient, agent: SokobanReActAgent, config: Dict, instance_num: int
) -> Dict[str, Any]:
    """Run a single Sokoban episode and return episode metrics."""
    try:
        # Create environment
        create_resp = await client.post(f"/env/Sokoban/initialize", json={"initial_state": config})

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
        print(f"\n  Instance {instance_num}: Starting puzzle")
        print(f"  Initial state:")
        print(f"  {formatted_obs}")

        # Track episode metrics
        steps_taken = 0
        max_steps = config.get("max_steps", 120)

        # Run episode
        for turn in range(agent.max_turns):
            # Get agent decision
            action = await agent.decide(formatted_obs, agent.get_system_message(), turn)

            # DEBUG: Print agent decision
            print(
                f"  Turn {turn + 1}: Agent chose '{action['parameters']['action']}' - {action['parameters'].get('reasoning', 'no reasoning')}"
            )

            # Check for termination
            if action["name"] == "terminate":
                print(
                    f"  Agent terminated: {action['parameters'].get('reason', 'no reason given')}"
                )
                break

            # Execute action in environment
            action_name = action["parameters"]["action"]

            # Convert action string to integer (Sokoban expects integers)
            if action_name in ACTION_STRING_TO_INT:
                action_int = ACTION_STRING_TO_INT[action_name]
            else:
                print(f"  ❌ Unknown action '{action_name}', using no-op")
                action_int = 0  # Default to "no operation"

            step_resp = await client.post(
                f"/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": action_int}}]
                    },
                },
            )

            if step_resp.status_code != 200:
                print(f"  ❌ Step failed: {step_resp.status_code}: {step_resp.text}")
                break

            obs = step_resp.json()["observation"]
            formatted_obs = agent.format_observation(obs)

            # DEBUG: Print state after action
            print(f"  After action:")
            print(f"  {formatted_obs}")

            # Update history
            agent.history.append(f"{action_name}: {action['parameters'].get('reasoning', '')[:50]}")

            # Track steps
            steps_taken = obs.get("steps_taken", steps_taken + 1)

            # Check if game is won
            boxes_on_target = obs.get("boxes_on_target", 0)
            num_boxes = obs.get("num_boxes", 0)
            terminated = obs.get("terminated", False)

            if terminated and boxes_on_target == num_boxes:
                print(
                    f"  ✅ Instance {instance_num}: SUCCESS! All boxes on target in {steps_taken} steps"
                )

                # Calculate eval metric and rubric
                eval_metric = 1.0

                # Create rubric - we'll estimate optimal solution as a fraction of max_steps
                # This is a rough estimate since we don't have actual optimal solutions
                estimated_optimal = max(num_boxes * 3, 10)  # Rough estimate
                step_efficiency = min(1.0, estimated_optimal / max(steps_taken, 1))

                rubric = {
                    "solved": 1.0,
                    "step_efficiency": step_efficiency,
                    "boxes_placed": float(boxes_on_target) / max(num_boxes, 1),
                    "completed_in_time": 1.0 if steps_taken <= max_steps else 0.0,
                }

                await client.post(f"/env/Sokoban/terminate", json={"env_id": env_id})
                return {
                    "eval_metric": eval_metric,
                    "rubric": rubric,
                    "steps_taken": steps_taken,
                    "boxes_on_target": boxes_on_target,
                    "num_boxes": num_boxes,
                    "solved": True,
                    "error": False,
                }

            if terminated:
                print(
                    f"  ❌ Instance {instance_num}: Game terminated without success (boxes: {boxes_on_target}/{num_boxes})"
                )
                break

        print(f"  ❌ Instance {instance_num}: Failed to solve in {agent.max_turns} turns")

        # Calculate eval metric and rubric for failed episode
        eval_metric = 0.0
        rubric = {
            "solved": 0.0,
            "step_efficiency": 0.0,
            "boxes_placed": float(boxes_on_target) / max(num_boxes, 1),
            "completed_in_time": 0.0,
        }

        # Cleanup
        await client.post(f"/env/Sokoban/terminate", json={"env_id": env_id})
        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "steps_taken": steps_taken,
            "boxes_on_target": boxes_on_target,
            "num_boxes": num_boxes,
            "solved": False,
            "error": False,
        }

    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback

        traceback.print_exc()
        return {"eval_metric": 0.0, "rubric": {}, "error": True}


# --- Batch Evaluation ---
async def evaluate_sokoban_batch() -> Dict[str, Any]:
    """Evaluate Sokoban agent on multiple easy instances."""
    print(f"🎯 Evaluating Sokoban on {NUM_INSTANCES} easy instances...")

    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)

    # Get easy task instances using the taskset system
    from synth_ai.environments.examples.sokoban.taskset import create_task_instance_from_seed

    easy_task_instances = []
    task_debug_info = []

    for seed in range(NUM_INSTANCES):
        try:
            print(f"  🔍 Creating task instance for seed {seed}...")
            task_instance = await create_task_instance_from_seed(DIFFICULTY, seed)
            easy_task_instances.append(task_instance)

            # Extract debug info
            task_id = getattr(task_instance, "id", "unknown")
            metadata = getattr(task_instance, "metadata", {})
            initial_snapshot = getattr(task_instance, "initial_engine_snapshot", {})

            debug_info = {
                "seed": seed,
                "task_id": task_id,
                "metadata": metadata,
                "room_state_hash": hash(str(initial_snapshot.get("room_state", []))),
                "room_fixed_hash": hash(str(initial_snapshot.get("room_fixed", []))),
                "num_boxes": initial_snapshot.get("num_boxes", 0),
                "max_steps": initial_snapshot.get("max_steps", 0),
            }
            task_debug_info.append(debug_info)

            print(
                f"    ✅ Seed {seed}: task_id={task_id}, room_state_hash={debug_info['room_state_hash']}"
            )

        except Exception as e:
            print(f"  ⚠️  Failed to get task instance for seed {seed}: {e}")
            continue

    print(f"  📝 Generated {len(easy_task_instances)} {DIFFICULTY} task instances from seeds 0,1,2")

    # Print debug summary
    print(f"  🔍 Task Debug Summary:")
    for info in task_debug_info:
        print(
            f"    Seed {info['seed']}: ID={info['task_id']}, StateHash={info['room_state_hash']}, FixedHash={info['room_fixed_hash']}"
        )

    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=30.0) as client:
        tasks = []
        for i, task_instance in enumerate(easy_task_instances):
            agent = SokobanReActAgent(llm, max_turns=MAX_TURNS, verbose=False)

            # Extract configuration from task instance
            config = {
                "dim_room": list(task_instance.metadata.dim_room),
                "max_steps": task_instance.metadata.max_steps,
                "num_boxes": task_instance.metadata.num_boxes,
                "room_fixed": task_instance.initial_engine_snapshot["room_fixed"],
                "room_state": task_instance.initial_engine_snapshot["room_state"],
                "boxes_on_target": task_instance.initial_engine_snapshot.get("boxes_on_target", 0),
            }

            tasks.append(run_single_episode(client, agent, config, i + 1))

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

        # Calculate mean rubric values
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(r["rubric"].keys())

        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)

        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "mean_rubric": mean_rubric,
            "num_episodes": len(valid_results),
        }


async def main():
    """Run Sokoban evaluation."""
    print(f"🎮 Sokoban ReAct Agent Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Service: {SERVICE_BASE_URL}")
    print(f"Instances: {NUM_INSTANCES}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Test service health
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=10.0) as client:
        try:
            health_resp = await client.get("/health")
            health_data = health_resp.json()

            if "Sokoban" not in health_data.get("supported_environments", []):
                print("❌ Sokoban not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return

    # Run evaluation
    try:
        results = await evaluate_sokoban_batch()

        print("\n" + "=" * 80)
        print("🏆 FINAL SOKOBAN EVALUATION RESULTS")
        print("=" * 80)

        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.1f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")

        # Print rubric results
        print(f"\n🎯 RUBRIC RESULTS:")
        if results["mean_rubric"]:
            for metric, score in sorted(results["mean_rubric"].items()):
                print(f"  {metric}: {score:.2f}")
        else:
            print("  No rubric data available")

        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        if results["mean_eval_metric"] > 0.5:
            print("🎉 Excellent performance!")
        elif results["mean_eval_metric"] > 0.3:
            print("✅ Good performance!")
        elif results["mean_eval_metric"] > 0.1:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - needs improvement")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
