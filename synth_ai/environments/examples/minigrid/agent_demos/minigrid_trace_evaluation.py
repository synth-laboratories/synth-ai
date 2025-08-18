#!/usr/bin/env python3
"""
Simple MiniGrid evaluation script to generate traces.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import base64
import io

import gymnasium as gym
import minigrid
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image


# Environment setup
def create_minigrid_env(env_name="MiniGrid-Empty-6x6-v0"):
    """Create a MiniGrid environment with image observations."""
    env = gym.make(env_name)
    # Wrap to get RGB image observations
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env


def image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    # Convert to PIL Image
    img = Image.fromarray(image_array.astype(np.uint8))
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def get_action_name(action_idx):
    """Map action index to name."""
    action_names = {
        0: "left",
        1: "right",
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done",
    }
    return action_names.get(action_idx, f"action_{action_idx}")


async def run_simple_minigrid_eval(
    model_name="simple-agent",
    env_name="MiniGrid-Empty-6x6-v0",
    num_episodes=3,
    max_steps=50,
):
    """Run a simple evaluation to generate MiniGrid traces."""

    print(f"\n🎮 Running MiniGrid Evaluation")
    print(f"   Environment: {env_name}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps: {max_steps}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{int(datetime.now().timestamp())}"
    output_dir = Path(f"src/evals/minigrid/{run_id}")
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")

        # Create environment
        env = create_minigrid_env(env_name)
        obs, info = env.reset()

        # Initialize trace
        trace_id = str(uuid.uuid4())
        trace_data = {
            "trace": {
                "metadata": {
                    "model_name": model_name,
                    "env_name": env_name,
                    "difficulty": "easy",
                    "seed": episode,
                    "max_steps": max_steps,
                },
                "partition": [],
            },
            "dataset": {"reward_signals": []},
        }

        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Simple policy: random actions with bias towards forward
            if np.random.random() < 0.6:
                action = 2  # forward
            else:
                action = env.action_space.sample()

            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Create partition for this step
            partition = {
                "events": [
                    {
                        "environment_compute_steps": [
                            {
                                "compute_output": [
                                    {
                                        "outputs": {
                                            "observation": {
                                                "mission": getattr(
                                                    env.unwrapped,
                                                    "mission",
                                                    "Reach the goal",
                                                ),
                                                "image_base64": image_to_base64(
                                                    obs
                                                    if isinstance(obs, np.ndarray)
                                                    else obs["image"]
                                                ),
                                            },
                                            "action": action,
                                            "reward": float(reward),
                                            "terminated": terminated,
                                            "truncated": truncated,
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

            trace_data["trace"]["partition"].append(partition)

            obs = next_obs
            step += 1

            if done and reward > 0:
                print(f"   ✅ Success! Reached goal in {step} steps")

        if not done:
            print(f"   ⏰ Timeout after {step} steps")

        # Update trace metadata
        trace_data["trace"]["metadata"]["success"] = reward > 0
        trace_data["trace"]["metadata"]["num_steps"] = step
        trace_data["dataset"]["reward_signals"].append({"reward": float(total_reward)})

        # Save trace
        trace_file = traces_dir / f"minigrid_trace_{trace_id}.json"
        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)

        results.append(
            {
                "trace_id": trace_id,
                "success": reward > 0,
                "steps": step,
                "total_reward": total_reward,
            }
        )

        print(f"   💾 Saved trace: {trace_file.name}")

    # Save evaluation summary
    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "environment": env_name,
        "model_name": model_name,
        "num_episodes": num_episodes,
        "results": results,
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
        "avg_steps": sum(r["steps"] for r in results) / len(results),
        "models_evaluated": [model_name],
        "difficulties_evaluated": ["easy"],
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average steps: {summary['avg_steps']:.1f}")
    print(f"   Output directory: {output_dir}")

    return summary


if __name__ == "__main__":
    # Run evaluation
    asyncio.run(
        run_simple_minigrid_eval(env_name="MiniGrid-Empty-6x6-v0", num_episodes=3, max_steps=30)
    )
