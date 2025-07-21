from __future__ import annotations

import asyncio
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from synth_ai.zyk import LM
from synth_ai.environments.examples.tictactoe.environment import TicTacToeEnvironment
from synth_ai.environments.examples.tictactoe.taskset import create_tictactoe_taskset


class TicTacToeActionInput(BaseModel):
    action: str  # "A1", "B2", etc.


class TerminateArgs(BaseModel):
    reason: Optional[str] = None


class TicTacToeReActAgent:
    def __init__(self, llm, max_turns: int = 9):
        self.llm = llm
        self.max_turns = max_turns
        self.history = []
        self.system_name = "tictactoe-react"

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "tictactoe_interact",
                    "description": "Place your mark in a cell. Valid cells are A1-A3, B1-B3, C1-C3.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The cell coordinate (e.g. A1, B2, C3)",
                            }
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "End the game if it's completed (win/draw/loss) or no valid moves remain.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for terminating",
                            }
                        },
                    },
                },
            },
        ]

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM consumption"""
        lines = []

        if "board_text" in obs:
            lines.append("Current Board:")
            lines.append(obs["board_text"])

        if "current_player" in obs:
            lines.append(f"\nCurrent Player: {obs['current_player']}")

        if "last_move" in obs and obs["last_move"]:
            lines.append(f"Last Move: {obs['last_move']}")

        if "move_count" in obs:
            lines.append(f"Move Count: {obs['move_count']}/9")

        if "winner" in obs and obs["winner"]:
            lines.append(f"\nGame Result: {obs['winner']}")

        if "reward_last" in obs and obs["reward_last"] != 0:
            lines.append(f"Reward: {obs['reward_last']}")

        if "error" in obs:
            lines.append(f"\nError: {obs['error']}")

        return "\n".join(lines)

    async def decide(self, obs: str) -> Dict[str, Any]:
        """Get LLM decision based on observation"""
        # Add observation to history
        self.history.append({"role": "user", "content": obs})

        # Create prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are playing TicTacToe. Analyze the board state and make strategic moves. "
                    "Use tictactoe_interact to place your mark, and terminate when the game ends."
                ),
            }
        ] + self.history

        # Get LLM response
        response = await self.llm.respond_async(
            system_message=messages[0]["content"], user_message=obs, tools=self.tools
        )

        # Parse tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]

            # Handle different response structures
            if hasattr(tool_call, "function"):
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
            elif isinstance(tool_call, dict):
                if "function" in tool_call:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                else:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("arguments", {})
            else:
                return {
                    "name": "terminate",
                    "parameters": {"reason": "Unexpected tool call format"},
                }

            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)

            return {"name": tool_name, "parameters": tool_args}
        else:
            # Default to a terminate if no tool call
            return {"name": "terminate", "parameters": {"reason": "No valid action"}}

    async def run_episode(self, env: TicTacToeEnvironment) -> Dict[str, Any]:
        """Run a single episode"""
        # Initialize
        obs = await env.initialize()
        formatted_obs = self._format_observation(obs)

        # Reset history
        self.history = []

        # Track episode data
        episode_data = {
            "turns": 0,
            "winner": None,
            "final_reward": 0.0,
            "terminated_correctly": False,
            "moves": [],
        }

        # Main game loop
        for turn in range(self.max_turns):
            episode_data["turns"] = turn + 1

            # Get agent decision
            action = await self.decide(formatted_obs)

            # Check for termination
            if action["name"] == "terminate":
                final_obs = await env.checkpoint()
                episode_data["terminated_correctly"] = True
                episode_data["winner"] = final_obs.get("winner_final")
                episode_data["final_reward"] = final_obs.get("total_reward", 0.0)
                break

            # Execute action
            tool_call = {"name": "interact", "parameters": action["parameters"]}

            obs = await env.step(tool_call)
            formatted_obs = self._format_observation(obs)

            # Track move
            if "last_move" in obs:
                episode_data["moves"].append(obs["last_move"])

            # Check if game ended
            if obs.get("terminated", False) or obs.get("winner"):
                episode_data["winner"] = obs.get("winner")
                episode_data["final_reward"] = obs.get("total_reward", 0.0)
                episode_data["terminated_correctly"] = False  # Should have used terminate
                break

        return episode_data


async def eval_react_tictactoe(
    model_name: str = "gpt-4.1-mini", num_episodes: int = 10
) -> List[Dict[str, Any]]:
    """Run ReAct agent evaluation on TicTacToe taskset"""

    # Load taskset
    taskset = await create_tictactoe_taskset()

    # Initialize LLM and agent
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.7)
    agent = TicTacToeReActAgent(llm)

    # Run episodes
    results = []
    for i, instance in enumerate(taskset.instances[:num_episodes]):
        print(f"\nRunning episode {i + 1}/{num_episodes}...")

        # Create environment
        env = TicTacToeEnvironment(instance)

        # Run episode
        result = await agent.run_episode(env)
        result["task_metadata"] = {
            "starting_player": instance.metadata.starting_player,
            "opening_moves": instance.metadata.opening_moves,
            "optimal_outcome": instance.metadata.optimal_outcome,
            "position_complexity": instance.metadata.position_complexity,
        }

        results.append(result)

        # Print result
        print(
            f"  Result: {result['winner']}, Moves: {len(result['moves'])}, Reward: {result['final_reward']}"
        )

    return results


async def test_react_agent_tictactoe():
    """Test function for pytest compatibility"""
    results = await eval_react_tictactoe(num_episodes=5)

    # Basic assertions
    assert len(results) > 0
    assert all("winner" in r for r in results)
    assert all("turns" in r for r in results)

    # Calculate statistics
    total_games = len(results)
    wins = sum(1 for r in results if r["winner"] == "X")
    draws = sum(1 for r in results if r["winner"] == "draw")
    losses = sum(1 for r in results if r["winner"] == "O")

    win_rate = wins / total_games if total_games > 0 else 0

    print(f"\n=== TicTacToe ReAct Agent Results ===")
    print(f"Total Games: {total_games}")
    print(f"Wins (X): {wins} ({win_rate:.1%})")
    print(f"Draws: {draws} ({draws / total_games:.1%})")
    print(f"Losses (O): {losses} ({losses / total_games:.1%})")
    print(f"Average Moves: {sum(len(r['moves']) for r in results) / total_games:.1f}")

    # Check that agent can at least draw in simple positions
    simple_games = [r for r in results if r["task_metadata"]["position_complexity"] == 0]
    if simple_games:
        simple_non_losses = sum(1 for r in simple_games if r["winner"] != "O")
        print(f"Non-loss rate in fresh games: {simple_non_losses / len(simple_games):.1%}")


if __name__ == "__main__":
    asyncio.run(test_react_agent_tictactoe())
