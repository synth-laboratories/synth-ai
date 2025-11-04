"""Warming Up to RL baseline for Crafter.

This baseline demonstrates how to evaluate an LLM agent on the Crafter survival game
without requiring a deployed task app. This is the recommended starting point for coding
agents to get a baseline score before making changes.

Quick Start:
    # Run a quick 3-task baseline
    uvx synth-ai baseline warming_up_to_rl --split train --seeds 0,1,2

    # Full train evaluation
    uvx synth-ai baseline warming_up_to_rl --split train

    # Compare models
    uvx synth-ai baseline warming_up_to_rl --model groq:openai/gpt-oss-20b
"""

from __future__ import annotations

import json
from typing import Any

try:
    import crafter
    CRAFTER_AVAILABLE = True
except ImportError:
    CRAFTER_AVAILABLE = False

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.types import EventReward, OutcomeReward


class CrafterRunner(BaselineTaskRunner):
    """Task runner for Crafter environment."""

    def __init__(self, policy_config: dict[str, Any], env_config: dict[str, Any]):
        super().__init__(policy_config, env_config)
        self.max_steps = env_config.get("max_steps", 1000)

    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Crafter episode."""
        if not CRAFTER_AVAILABLE:
            raise ImportError(
                "Crafter not installed. Install with: pip install crafter"
            )

        # Create environment
        env = crafter.Env()
        env.reset()

        # Initialize tracking
        event_rewards: list[EventReward] = []
        achievements = {}
        step_count = 0

        # Get model configuration
        from synth_ai.inference.client import InferenceClient

        client = InferenceClient()
        model = self.policy_config.get("model", "gpt-4o-mini")
        temperature = self.policy_config.get("temperature", 0.7)

        # Define action tool
        actions = [
            "noop", "move_left", "move_right", "move_up", "move_down",
            "do", "sleep", "place_stone", "place_table", "place_furnace",
            "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
            "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
            "make_iron_sword"
        ]

        action_tool = {
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Take an action in the Crafter world",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": actions,
                            "description": f"Action to take. Available: {', '.join(actions)}",
                        }
                    },
                    "required": ["action"],
                },
            },
        }

        # Run episode
        done = False
        while not done and step_count < self.max_steps:
            # Get observation (would include visual state in full implementation)
            obs_str = f"Crafter Step {step_count}\n"
            obs_str += f"Current achievements: {achievements}\n"
            obs_str += "What action should you take to survive and progress?"

            # Get action from model
            try:
                response = await client.generate(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at survival games. Use the take_action tool to survive and achieve goals in Crafter.",
                        },
                        {"role": "user", "content": obs_str},
                    ],
                    tools=[action_tool],
                    temperature=temperature,
                    max_tokens=100,
                )

                # Extract action
                action_name = "noop"
                if response.get("tool_calls"):
                    tool_call = response["tool_calls"][0]
                    args = json.loads(tool_call["function"]["arguments"])
                    action_name = args.get("action", "noop")

                action_idx = actions.index(action_name) if action_name in actions else 0

                # Take step
                obs, reward, done, info = env.step(action_idx)

                # Update achievements
                if "achievements" in info:
                    achievements.update(info["achievements"])

                # Track rewards
                if reward > 0:
                    event_rewards.append(
                        EventReward(
                            event_id=f"step_{step_count}",
                            reward=reward,
                            metadata={"action": action_name, "achievements": achievements.copy()},
                        )
                    )

                step_count += 1

            except Exception as e:
                done = True
                break

        # Calculate outcome reward based on achievements
        total_achievements = sum(achievements.values())
        success = total_achievements >= 3  # At least 3 achievements

        return TaskResult(
            success=success,
            outcome_reward=OutcomeReward(
                reward=float(total_achievements),
                metadata={
                    "steps": step_count,
                    "achievements": achievements,
                    "seed": seed,
                },
            ),
            event_rewards=event_rewards,
            total_steps=step_count,
            metadata={"achievements": achievements},
        )


# Define baseline configuration (only if Crafter is available)
if CRAFTER_AVAILABLE:
    warming_up_to_rl_baseline = BaselineConfig(
        baseline_id="warming_up_to_rl",
        name="Warming Up to RL - Crafter",
        description="Crafter survival game baseline for comparing agent performance on RL tasks",
        task_runner=CrafterRunner,
        splits={
            "train": DataSplit(name="train", seeds=list(range(20))),
            "val": DataSplit(name="val", seeds=list(range(20, 25))),
            "test": DataSplit(name="test", seeds=list(range(25, 30))),
        },
        default_policy_config={
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
        default_env_config={
            "max_steps": 1000,
        },
        tags=["rl", "survival", "achievements", "blog-post"],
    )
