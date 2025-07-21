#!/usr/bin/env python3
"""
Test script to run ReAct agents against MiniGrid environment on synth service (port 8901)
Tests on multiple easy MiniGrid instances with enhanced debugging
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
MODEL_NAME = "o3"
NUM_INSTANCES = 1
MAX_TURNS = 20
DIFFICULTY = "ultra_easy"


# --- Tool Definitions ---
class NavigationActionArgs(BaseModel):
    """Arguments for navigation actions."""

    action: str = Field(
        description="The action to take: left, right, forward, pickup, drop, toggle, done"
    )
    reasoning: str = Field(description="Brief explanation of why this action was chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


class NavigationActionTool(BaseTool):
    """Tool for performing an action in the MiniGrid environment."""

    name: str = "navigation_action"
    arguments: type[BaseModel] = NavigationActionArgs
    description: str = "Perform a navigation action in the MiniGrid environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""

    def __init__(self, llm: LM, max_turns: int = 15, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"

        # Define tools in OpenAI format (like Enron agent)
        self.tools = [
            NavigationActionTool(),
            TerminateTool(),
        ]

    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision based on observation."""
        # Create conversation context
        context = f"Turn {turn + 1}/{self.max_turns}\n\n{obs}"

        # Generate response using LLM (same pattern as Crafter)
        response_obj = await self.llm.respond_async(
            system_message=system_message, user_message=context, tools=self.tools
        )

        tool_calls = response_obj.tool_calls

        # Handle case where tool_calls is None or empty (graceful fallback)
        if not tool_calls:
            if self.verbose:
                print(f"[WARNING] No tool calls returned by LLM, using default action")
            return {
                "name": "navigation_action",
                "parameters": {
                    "action": "forward",
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


# --- MiniGrid ReAct Agent ---
class MiniGridReActAgent(BaseReActAgent):
    """ReAct agent for MiniGrid environment."""

    def __init__(self, llm: LM, max_turns: int = 15, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "minigrid-react-agent"

    def get_system_message(self) -> str:
        return """You are navigating a MiniGrid environment. Your goal is to reach the goal (G) to complete the mission successfully.

ACTIONS: 
- "left": turn left (counter-clockwise)
- "right": turn right (clockwise)  
- "forward": move forward one step
- "pickup": pick up object in front of you
- "drop": drop carried object
- "toggle": open/close door or interact with object
- "done": complete mission when you reach the goal

SYMBOLS: 
- # = wall (blocks movement)
- . = empty space (can move through)
- G = goal (your destination)
- K = key (pick up to unlock doors)
- D = door (may need key to open)
- L = lava (avoid - will end mission)
- @ = you (your current position)

STRATEGY:
1. Analyze the grid layout to understand the environment
2. Plan a path to reach the goal (G)
3. Navigate systematically - turn to face the right direction, then move forward
4. Pick up keys (K) before trying to open doors (D)
5. Use "toggle" to open doors when you have the key
6. Avoid lava (L) at all costs
7. Use "done" when you reach the goal

IMPORTANT: You can only see a limited view around you. Move and explore to discover the full environment. Be systematic in your exploration."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for MiniGrid."""
        parts = []

        if "grid" in obs:
            parts.append(f"Grid view:\n{obs['grid']}")
        elif "observation" in obs:
            parts.append(f"Observation:\n{obs['observation']}")

        if "direction" in obs:
            parts.append(f"Facing: {obs['direction']}")

        if "carrying" in obs and obs["carrying"]:
            parts.append(f"Carrying: {obs['carrying']}")

        if "step_count" in obs:
            parts.append(f"Steps: {obs['step_count']}")

        if "mission" in obs:
            parts.append(f"Mission: {obs['mission']}")

        # Add more possible observation fields
        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")

        if "success" in obs:
            parts.append(f"Success: {obs['success']}")

        if "reward_last" in obs:
            parts.append(f"Last reward: {obs['reward_last']}")

        return "\n".join(parts) if parts else "No formatted observation available"


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient, agent: MiniGridReActAgent, task_instance, instance_num: int
) -> bool:
    """Run a single MiniGrid episode and return success status."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/MiniGrid/initialize", json={"task_instance": await task_instance.serialize()}
        )

        if create_resp.status_code != 200:
            print(
                f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}"
            )
            return False

        env_id = create_resp.json()["env_id"]

        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)

        # DEBUG: Print initial state
        print(f"\n  Instance {instance_num}: Starting MiniGrid mission")
        print(f"  Environment: {task_instance.metadata.env_name}")
        print(f"  Mission: {task_instance.impetus.instructions[:100]}...")
        print(f"  Initial observation:")
        print(f"    {formatted_obs}")

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

            step_resp = await client.post(
                f"/env/MiniGrid/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "minigrid_act", "args": {"action": action_name}}]
                    },
                },
            )

            if step_resp.status_code != 200:
                print(f"  ❌ Step failed: {step_resp.status_code}: {step_resp.text}")
                break

            obs = step_resp.json()["observation"]
            formatted_obs = agent.format_observation(obs)

            # DEBUG: Print state after action
            print(f"  After action: {formatted_obs}")

            # Update history
            agent.history.append(f"{action_name}: {action['parameters'].get('reasoning', '')[:50]}")

            # Check if goal is reached
            terminated = obs.get("terminated", False)
            success = obs.get("success", False)
            reward_last = obs.get("reward_last", 0.0)

            # MiniGrid success is typically indicated by positive reward when terminated
            # Success reward is usually close to 1.0 (1.0 - step_penalties)
            actual_success = terminated and reward_last > 0.1  # Threshold for success reward

            if terminated and actual_success:
                print(
                    f"  ✅ Instance {instance_num}: SUCCESS! Mission completed in {turn + 1} turns (reward: {reward_last:.3f})"
                )
                await client.post(f"/env/MiniGrid/terminate", json={"env_id": env_id})
                return True

            if terminated:
                print(
                    f"  ❌ Instance {instance_num}: Terminated without success (success field: {success}, reward: {reward_last:.3f})"
                )
                break

        print(
            f"  ❌ Instance {instance_num}: Failed to complete mission in {agent.max_turns} turns"
        )

        # Cleanup
        await client.post(f"/env/MiniGrid/terminate", json={"env_id": env_id})
        return False

    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback

        traceback.print_exc()
        return False


# --- Batch Evaluation ---
async def evaluate_minigrid_batch() -> float:
    """Evaluate MiniGrid agent on multiple easy instances."""
    print(f"🎯 Evaluating MiniGrid on {NUM_INSTANCES} easy instances...")

    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)

    # Get easy task instances using the taskset system
    from synth_ai.environments.examples.minigrid.taskset import create_minigrid_task_from_seed

    easy_task_instances = []
    for seed in range(NUM_INSTANCES):
        try:
            task_instance = await create_minigrid_task_from_seed(DIFFICULTY, seed)
            easy_task_instances.append(task_instance)
        except Exception as e:
            print(f"  ⚠️  Failed to get task instance for seed {seed}: {e}")
            continue

    print(
        f"  📝 Generated {len(easy_task_instances)} {DIFFICULTY} task instances from seeds 0-{NUM_INSTANCES - 1}"
    )

    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=30.0) as client:
        tasks = []
        for i, task_instance in enumerate(easy_task_instances):
            agent = MiniGridReActAgent(llm, max_turns=MAX_TURNS, verbose=False)
            tasks.append(run_single_episode(client, agent, task_instance, i + 1))

        results = await asyncio.gather(*tasks)
        success_count = sum(results)
        success_rate = success_count / len(easy_task_instances)

        print(
            f"  📊 MiniGrid Results: {success_count}/{len(easy_task_instances)} solved ({success_rate:.1%})"
        )
        return success_rate


async def main():
    """Run MiniGrid evaluation."""
    print(f"🎮 MiniGrid ReAct Agent Evaluation")
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

            if "MiniGrid" not in health_data.get("supported_environments", []):
                print("❌ MiniGrid not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return

    # Run evaluation
    try:
        success_rate = await evaluate_minigrid_batch()

        print("\n" + "=" * 50)
        print("🏆 FINAL MINIGRID RESULTS")
        print("=" * 50)
        print(f"Success Rate: {success_rate:.1%}")

        if success_rate > 0.5:
            print("🎉 Excellent performance!")
        elif success_rate > 0.3:
            print("✅ Good performance!")
        elif success_rate > 0.1:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - needs improvement")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
