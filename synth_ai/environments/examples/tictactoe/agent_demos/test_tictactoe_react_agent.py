#!/usr/bin/env python3
"""
Test script to run ReAct agents against TicTacToe environment on synth service (port 8901)
Tests on multiple TicTacToe instances with random opponent moves
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool


# --- Service Configuration ---
SERVICE_BASE_URL = "http://localhost:8901"
MODEL_NAME = "o3"
NUM_INSTANCES = 5
MAX_TURNS = 9  # TicTacToe has at most 9 moves
DIFFICULTY = "random"


# --- Tool Definitions ---
class TicTacToeActionArgs(BaseModel):
    """Arguments for tictactoe actions."""

    action: str = Field(description="Cell coordinate (e.g., A1, B2, C3)")
    reasoning: str = Field(description="Brief explanation of why this move was chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


class TicTacToeActionTool(BaseTool):
    """Tool for performing a move in the TicTacToe environment."""

    name: str = "tictactoe_interact"
    arguments: type[BaseModel] = TicTacToeActionArgs
    description: str = "Place your mark in a cell. Valid cells are A1-A3, B1-B3, C1-C3."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the game when finished or no progress can be made."


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""

    def __init__(self, llm: LM, max_turns: int = 9, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"

        # Define tools in OpenAI format
        self.tools = [
            TicTacToeActionTool(),
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
                "name": "tictactoe_interact",
                "parameters": {
                    "action": "B2",  # Center is usually a safe default
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


# --- TicTacToe ReAct Agent ---
class TicTacToeReActAgent(BaseReActAgent):
    """ReAct agent for TicTacToe environment."""

    def __init__(self, llm: LM, max_turns: int = 9, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "tictactoe-react-agent"

    def get_system_message(self) -> str:
        return """You are playing TicTacToe against a random opponent. Your goal is to win or at least force a draw.

CRITICAL RULES:
- You play on a 3x3 grid with cells labeled A1-A3, B1-B3, C1-C3
- You MUST ONLY choose from cells listed as "Available" in the observation
- NEVER choose cells listed as "Occupied" - this will cause an illegal move and immediate loss
- Get three of your marks in a row (horizontally, vertically, or diagonally) to win
- If no one gets three in a row and the board is full, it's a draw

STRATEGY:
1. Try to get three in a row to win
2. Block your opponent from getting three in a row
3. Take center (B2) if available - it's usually the best opening
4. Take corners if center is not available
5. Avoid giving opponent easy wins

COORDINATE SYSTEM:
  1 2 3
A . . .
B . . .
C . . .

IMPORTANT: Always check the "Available" cells list in the observation and ONLY choose from those cells. Choosing an occupied cell will result in an illegal move and automatic loss."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for TicTacToe with enhanced clarity."""
        parts = []

        if "board_text" in obs:
            parts.append("Current Board:")
            parts.append(obs["board_text"])

            # Add explicit cell status for clarity
            board_lines = obs["board_text"].strip().split("\n")
            if len(board_lines) >= 4:
                parts.append("\nCell Status:")
                occupied = []
                available = []

                # Parse board more carefully - the display format is:
                #   A B C
                # 1 . . .
                # 2 . X .
                # 3 . . .
                # Where A,B,C are COLUMNS and 1,2,3 are ROWS
                # But our coordinate system is A1-A3, B1-B3, C1-C3 where:
                # - A,B,C are ROWS
                # - 1,2,3 are COLUMNS

                for i, line in enumerate(board_lines[1:4]):  # Skip header
                    display_row = i + 1  # 1, 2, 3

                    if len(line) >= 2:
                        # Parse the line like "2 X X  "
                        cell_chars = line[2:] if len(line) > 2 else ""

                        # The board format uses space separators: "A B C" where positions are:
                        # Column A: position 0, space, Column B: position 2, space, Column C: position 4
                        column_positions = [0, 2, 4]  # Positions of A, B, C columns

                        # Extract characters from the 3 columns
                        for col_idx in range(3):
                            # Get the character at the correct position for this column
                            pos = column_positions[col_idx]
                            if pos < len(cell_chars):
                                cell = cell_chars[pos]
                            else:
                                cell = " "

                            # Convert display coordinates to our coordinate system:
                            # Display row 1 → our row A, Display row 2 → our row B, etc.
                            # Display col A → our col 1, Display col B → our col 2, etc.
                            our_row = ["A", "B", "C"][i]  # i is 0,1,2 → A,B,C
                            our_col = col_idx + 1  # 0,1,2 → 1,2,3
                            coord = f"{our_row}{our_col}"

                            if cell.strip() in ["X", "O"]:
                                occupied.append(f"{coord}={cell.strip()}")
                            else:
                                available.append(coord)

                if occupied:
                    parts.append(f"  Occupied: {', '.join(occupied)}")
                if available:
                    parts.append(f"  Available: {', '.join(available)}")

        if "current_player" in obs:
            parts.append(f"\nCurrent Player: {obs['current_player']}")

        if "last_move" in obs and obs["last_move"]:
            parts.append(f"Last Move: {obs['last_move']}")

        if "move_count" in obs:
            parts.append(f"Move Count: {obs['move_count']}/9")

        if "winner" in obs and obs["winner"]:
            parts.append(f"\nGame Result: {obs['winner']}")

        if "reward_last" in obs and obs["reward_last"] != 0:
            parts.append(f"Reward: {obs['reward_last']}")

        if "error" in obs:
            parts.append(f"\nError: {obs['error']}")

        return "\n".join(parts)


# Random opponent moves are now handled by the TicTacToe environment internally


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient, agent: TicTacToeReActAgent, task_instance, instance_num: int
) -> Dict[str, Any]:
    """Run a single TicTacToe episode and return episode metrics."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/TicTacToe/initialize", json={"task_instance": await task_instance.serialize()}
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
        print(f"\n  Instance {instance_num}: Starting TicTacToe game")
        print(f"  Agent plays as: {task_instance.metadata.starting_player}")
        print(f"  Opening moves: {task_instance.metadata.opening_moves}")
        print(f"  Initial observation:")
        print(f"    {formatted_obs}")

        # Track game state
        agent_player = task_instance.metadata.starting_player
        print(f"  DEBUG: agent_player = {agent_player}")

        # Run episode - TicTacToe handles opponent moves automatically
        for turn in range(agent.max_turns):
            # Check if game is already terminated
            if obs.get("terminated", False):
                break

            # Agent makes a move
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
                f"/env/TicTacToe/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": action_name}}]
                    },
                },
            )

            if step_resp.status_code != 200:
                print(f"  ❌ Step failed: {step_resp.status_code}: {step_resp.text}")
                break

            obs = step_resp.json()["observation"]
            formatted_obs = agent.format_observation(obs)

            # Update history
            agent.history.append(f"{action_name}: {action['parameters'].get('reasoning', '')[:50]}")

            # DEBUG: Print state after action
            print(f"  After move:")
            print(f"  {formatted_obs}")

            # Check if game ended
            terminated = obs.get("terminated", False)
            winner = obs.get("winner")

            if terminated:
                # DEBUG: Print evaluation details
                print(
                    f"  DEBUG: Game ended - winner='{winner}', agent_player='{agent_player}', winner==agent_player={winner == agent_player}"
                )

                # Calculate eval metric
                eval_metric = 0.0
                if winner == agent_player:
                    eval_metric = 1.0
                    print(f"  ✅ Instance {instance_num}: SUCCESS! Agent won as {agent_player}")
                elif winner == "draw":
                    eval_metric = 0.5
                    print(f"  ⚪ Instance {instance_num}: DRAW - acceptable result")
                else:
                    eval_metric = 0.0
                    print(f"  ❌ Instance {instance_num}: Agent lost to random opponent")

                await client.post(f"/env/TicTacToe/terminate", json={"env_id": env_id})
                return {
                    "eval_metric": eval_metric,
                    "rubric": {},  # No rubric for TicTacToe
                    "result": winner,
                    "agent_player": agent_player,
                    "error": False,
                }

        print(f"  ❌ Instance {instance_num}: Game didn't finish in {agent.max_turns} turns")

        # Cleanup
        await client.post(f"/env/TicTacToe/terminate", json={"env_id": env_id})
        return {"eval_metric": 0.0, "rubric": {}, "error": False}

    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback

        traceback.print_exc()
        return {"eval_metric": 0.0, "rubric": {}, "error": True}


# --- Batch Evaluation ---
async def evaluate_tictactoe_batch() -> Dict[str, Any]:
    """Evaluate TicTacToe agent on multiple instances."""
    print(f"🎯 Evaluating TicTacToe on {NUM_INSTANCES} instances with random opponent...")

    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)

    # Get task instances using the taskset system
    from synth_ai.environments.examples.tictactoe.taskset import create_tictactoe_taskset

    taskset = await create_tictactoe_taskset()
    task_instances = taskset.instances[:NUM_INSTANCES]

    print(f"  📝 Using {len(task_instances)} task instances from taskset")

    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=30.0) as client:
        tasks = []
        for i, task_instance in enumerate(task_instances):
            agent = TicTacToeReActAgent(llm, max_turns=MAX_TURNS, verbose=False)
            tasks.append(run_single_episode(client, agent, task_instance, i + 1))

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

        # Extract eval metrics (no rubric for TicTacToe)
        eval_metrics = [r["eval_metric"] for r in valid_results]
        mean_eval_metric = sum(eval_metrics) / len(eval_metrics)

        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "mean_rubric": {},  # No rubric for TicTacToe
            "num_episodes": len(valid_results),
        }


async def main():
    """Run TicTacToe evaluation."""
    print(f"🎮 TicTacToe ReAct Agent Evaluation")
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

            if "TicTacToe" not in health_data.get("supported_environments", []):
                print("❌ TicTacToe not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return

    # Run evaluation
    try:
        results = await evaluate_tictactoe_batch()

        print("\n" + "=" * 80)
        print("🏆 FINAL TICTACTOE EVALUATION RESULTS")
        print("=" * 80)

        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.1f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")

        # Print rubric results (none for TicTacToe)
        print(f"\n🎯 RUBRIC RESULTS:")
        print("  No rubric for TicTacToe")

        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        if results["mean_eval_metric"] > 0.8:
            print("🎉 Excellent performance against random opponent!")
        elif results["mean_eval_metric"] > 0.6:
            print("✅ Good performance!")
        elif results["mean_eval_metric"] > 0.4:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - struggling against random moves")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
