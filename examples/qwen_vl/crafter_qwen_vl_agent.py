#!/usr/bin/env python3
"""
Crafter agent using Qwen-VL models via synth-ai's hosted inference.

This demonstrates vision-language models (Qwen3-VL family) playing Crafter
with image observations. The CrafterPolicy automatically detects vision capability
from the model name and includes base64-encoded PNG frames in the prompt.

Requirements:
  - `SYNTH_API_KEY` environment variable (for synth-ai hosted inference)
  - synth-ai package with Crafter task app dependencies

Usage:
  uv run python examples/qwen_vl/crafter_qwen_vl_agent.py \
      --model Qwen/Qwen3-VL-8B-Instruct --seeds 10 --steps 20
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.task_apps.crafter.task_app.synth_envs_hosted.envs.crafter.environment import (
    CrafterEnvironmentWrapper,
)
from examples.task_apps.crafter.task_app.synth_envs_hosted.envs.crafter.policy import CrafterPolicy
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

# Import synth-ai inference client
try:
    from synth_ai.inference.client import InferenceClient
except ImportError:
    print("Error: synth-ai inference client not found. Make sure synth-ai is installed.")
    raise

DEFAULT_OUTPUT = Path("examples/qwen_vl/temp")
FRAME_SUBDIR = "qwen_vl_frames"


def _default_backend_base_url() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    return raw if raw.endswith("/api") else f"{raw}/api"


class EpisodeResult:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.steps_taken: int = 0
        self.achievements: set[str] = set()
        self.total_reward: float = 0.0
        self.tool_calls: int = 0

    def record_observation(self, observation: dict[str, Any]) -> None:
        obs = observation.get("observation") if isinstance(observation, dict) else None
        if not isinstance(obs, dict):
            return
        ach = obs.get("achievements_status")
        if isinstance(ach, dict):
            for name, unlocked in ach.items():
                if unlocked:
                    self.achievements.add(str(name))
        reward = obs.get("reward_last_step")
        if isinstance(reward, int | float):
            self.total_reward += float(reward)


def _ensure_synth_client() -> InferenceClient:
    """Initialize synth-ai inference client."""
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SYNTH_API_KEY must be set for synth-ai hosted inference. "
            "Get your key from https://synth-ai.com"
        )
    base_url = os.getenv("SYNTH_BASE_URL", _default_backend_base_url())
    return InferenceClient(base_url=base_url, api_key=api_key)


def _build_task_instance(seed: int) -> CrafterTaskInstance:
    """Create a Crafter task instance with specified seed."""
    impetus = Impetus(instructions="Explore, survive, and unlock achievements.")
    intent = Intent(
        rubric={"goal": "Maximise Crafter achievements."},
        gold_trajectories=None,
        gold_state_diff={},
    )
    metadata = CrafterTaskInstanceMetadata(
        difficulty="custom",
        seed=seed,
        num_trees_radius=0,
        num_cows_radius=0,
        num_hostiles_radius=0,
    )
    instance = CrafterTaskInstance(
        id=uuid4(),
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )
    setattr(instance, "config", {"seed": seed, "length": 256, "area": [64, 64]})
    return instance


def _decode_and_save_image(observation: dict[str, Any], path: Path) -> None:
    """Extract and save PNG frame from observation."""
    obs = observation.get("observation") if isinstance(observation, dict) else None
    if not isinstance(obs, dict):
        return
    base64_data = obs.get("observation_image_base64")
    if not isinstance(base64_data, str) or not base64_data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(Exception):
        path.write_bytes(base64.b64decode(base64_data))


async def _run_episode(
    *,
    seed: int,
    client: InferenceClient,
    model: str,
    max_steps: int,
    output_dir: Path,
    temperature: float,
) -> EpisodeResult:
    """Run a single Crafter episode with Qwen-VL."""
    task_instance = _build_task_instance(seed)
    env = CrafterClassicEnvironment(task_instance)
    wrapper = CrafterEnvironmentWrapper(env, seed=seed)
    
    # Policy will auto-detect vision from model name (qwen-vl and qwen3-vl tokens)
    policy = CrafterPolicy(inference_url="synth://inference", model=model)
    await policy.initialize({
        "use_tools": True,
        "model": model,
        "temperature": temperature,
        "max_tokens": 512,
    })

    episode_result = EpisodeResult(seed=seed)

    observation_packet = await wrapper.initialize()
    episode_result.record_observation(observation_packet)

    frames_root = output_dir / FRAME_SUBDIR / f"seed_{seed:04d}"
    _decode_and_save_image(observation_packet, frames_root / "step_000.png")

    for step_idx in range(max_steps):
        obs_dict = observation_packet.get("observation")
        if not isinstance(obs_dict, dict):
            break

        # Format observation text
        obs_text = policy._format_observation_for_llm(observation_packet)  # noqa: SLF001

        # Get tool calls from policy (it prepares the inference request internally)
        tool_calls, meta = await policy.step(
            observation_text=obs_text,
            metadata={"raw_observation": observation_packet},
        )
        if "inference_request" not in meta:
            break

        episode_result.steps_taken += 1
        inference_request = meta["inference_request"]

        # Call synth-ai hosted inference
        response = await client.create_chat_completion(
            model=model,
            messages=inference_request["messages"],
            temperature=temperature,
            max_tokens=512,
            tools=inference_request.get("tools"),
        )

        # Parse tool calls from response
        assistant_tool_calls = CrafterPolicy.parse_response_to_tool_calls(
            response,
            use_tools=policy.use_tools,
        )
        if not assistant_tool_calls:
            print(
                f"Seed {seed}: no tool calls returned by model; ending episode early at step {step_idx}."
            )
            break

        episode_result.tool_calls += len(assistant_tool_calls)

        # Extract assistant message
        assistant_message = response.get("choices", [{}])[0].get("message", {})
        assistant_text = assistant_message.get("content")

        # Execute action in environment
        env_response = await wrapper.step(assistant_tool_calls)
        if not isinstance(env_response, dict):
            raise RuntimeError(
                f"Unexpected environment response type: {type(env_response)!r}"
            )
        episode_result.record_observation(env_response)

        # Update policy history
        policy._append_assistant_turn(  # noqa: SLF001
            assistant_text,
            assistant_tool_calls,
            env_response,
        )

        # Save frame
        frame_path = frames_root / f"step_{step_idx + 1:03d}.png"
        _decode_and_save_image(env_response, frame_path)

        if env_response.get("done"):
            break
        observation_packet = env_response

    await wrapper.terminate()
    return episode_result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen-VL model name (e.g., Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds to evaluate")
    parser.add_argument("--steps", type=int, default=20, help="Max steps per seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for saved frames and summaries (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    client = _ensure_synth_client()
    results: list[EpisodeResult] = []

    seeds = list(range(args.seeds))
    print(f"Running {len(seeds)} Crafter episodes with model={args.model}")
    print(f"Using synth-ai hosted inference\n")

    for seed in seeds:
        result = await _run_episode(
            seed=seed,
            client=client,
            model=args.model,
            max_steps=args.steps,
            output_dir=args.output_dir,
            temperature=args.temperature,
        )
        results.append(result)
        print(
            f"Seed {seed:02d}: steps={result.steps_taken}, "
            f"achievements={len(result.achievements)}, "
            f"tool_calls={result.tool_calls}, reward≈{result.total_reward:.3f}"
        )

    summary = {
        "model": args.model,
        "provider": "synth-ai",
        "episodes": len(results),
        "mean_steps": round(
            sum(res.steps_taken for res in results) / max(len(results), 1), 2
        ),
        "mean_achievements": round(
            sum(len(res.achievements) for res in results) / max(len(results), 1), 2
        ),
        "total_tool_calls": sum(res.tool_calls for res in results),
        "output_dir": str(args.output_dir / FRAME_SUBDIR),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "qwen_vl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary")
    print("-------")
    print(json.dumps(summary, indent=2))
    print(f"\nFrames saved in: {summary['output_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
