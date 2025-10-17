#!/usr/bin/env python3
"""
Crafter agent that calls the OpenAI Chat Completions API with image + text prompts.

The harness mirrors the text-based agent workflow from `examples/warming_up_to_rl`:
  * Uses the Crafter policy to build prompts, maintain history, and parse tool calls.
  * Executes actions against the Synth Crafter environment (no HTTP task app required).
  * Persists every rendered frame to `examples/vlm/temp/` so you can inspect exactly
    what the VLM saw.

Requirements:
  - `OPENAI_API_KEY` environment variable.
  - `openai` Python package (installed via project dependencies).

Usage:
  uv run python examples/vlm/crafter_openai_vlm_agent.py \
      --model gpt-4o-mini-2024-07-18 --seeds 10 --steps 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.warming_up_to_rl.task_app.synth_envs_hosted.envs.crafter.environment import (
    CrafterEnvironmentWrapper,
)
from examples.warming_up_to_rl.task_app.synth_envs_hosted.envs.crafter.policy import CrafterPolicy
from openai import OpenAI
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

DEFAULT_OUTPUT = Path("examples/vlm/temp")
FRAME_SUBDIR = "openai_agent_frames"


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
        if isinstance(reward, (int, float)):
            self.total_reward += float(reward)


def _ensure_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for OpenAI calls")
    return OpenAI(api_key=api_key)


def _build_task_instance(seed: int) -> CrafterTaskInstance:
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
    instance.config = {"seed": seed, "length": 256, "area": [64, 64]}
    return instance


def _decode_and_save_image(observation: dict[str, Any], path: Path) -> None:
    obs = observation.get("observation") if isinstance(observation, dict) else None
    if not isinstance(obs, dict):
        return
    base64_data = obs.get("observation_image_base64")
    if not isinstance(base64_data, str) or not base64_data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(base64.b64decode(base64_data))
    except Exception:
        # Best-effort; corrupted frames should not halt rollout
        pass


def _normalise_openai_request(payload: dict[str, Any], model: str, temperature: float) -> dict[str, Any]:
    request = dict(payload)
    request["model"] = model
    request.setdefault("temperature", temperature)
    request.setdefault("max_tokens", 512)
    # Remove vendor-specific knobs unsupported by OpenAI
    request.pop("stop_after_tool_calls", None)
    request.pop("thinking_mode", None)
    request.pop("thinking_budget", None)
    max_completion = request.pop("max_completion_tokens", None)
    if max_completion is not None:
        request.setdefault("max_tokens", max_completion)
    return request


async def _run_episode(
    *,
    seed: int,
    client: OpenAI,
    model: str,
    max_steps: int,
    output_dir: Path,
    temperature: float,
) -> EpisodeResult:
    task_instance = _build_task_instance(seed)
    env = CrafterClassicEnvironment(task_instance)
    wrapper = CrafterEnvironmentWrapper(env, seed=seed)
    policy = CrafterPolicy(inference_url="openai://chat-completions", model=model)
    await policy.initialize({"use_tools": True, "model": model})

    episode_result = EpisodeResult(seed=seed)

    observation_packet = await wrapper.initialize()
    episode_result.record_observation(observation_packet)

    frames_root = output_dir / FRAME_SUBDIR / f"seed_{seed:04d}"
    _decode_and_save_image(observation_packet, frames_root / "step_000.png")

    for step_idx in range(max_steps):
        obs_dict = observation_packet.get("observation")
        if not isinstance(obs_dict, dict):
            break

        obs_text = policy._format_observation_for_llm(observation_packet)  # noqa: SLF001
        tool_calls, meta = await policy.step(
            observation_text=obs_text,
            metadata={"raw_observation": observation_packet},
        )
        if "inference_request" not in meta:
            break

        episode_result.steps_taken += 1
        inference_request = _normalise_openai_request(
            meta["inference_request"],
            model=model,
            temperature=temperature,
        )

        response = client.chat.completions.create(**inference_request)
        response_dict = response.model_dump()

        assistant_tool_calls = CrafterPolicy.parse_response_to_tool_calls(
            response_dict,
            use_tools=policy.use_tools,
        )
        if not assistant_tool_calls:
            print(
                f"Seed {seed}: no tool calls returned by model; ending episode early at step {step_idx}."
            )
            break

        episode_result.tool_calls += len(assistant_tool_calls)

        assistant_message = response_dict["choices"][0].get("message") or {}
        assistant_text = assistant_message.get("content")

        env_response = await wrapper.step(assistant_tool_calls)
        if not isinstance(env_response, dict):
            raise RuntimeError(
                f"Unexpected environment response type: {type(env_response)!r}"
            )
        episode_result.record_observation(env_response)

        policy._append_assistant_turn(  # noqa: SLF001
            assistant_text,
            assistant_tool_calls,
            env_response,
        )

        frame_path = frames_root / f"step_{step_idx + 1:03d}.png"
        _decode_and_save_image(env_response, frame_path)

        if env_response.get("done"):
            break
        observation_packet = env_response

    await wrapper.terminate()
    return episode_result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="OpenAI model id")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds to evaluate")
    parser.add_argument("--steps", type=int, default=10, help="Max steps per seed")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for saved frames and summaries (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    client = _ensure_client()
    results: list[EpisodeResult] = []

    seeds = list(range(args.seeds))
    print(f"Running {len(seeds)} Crafter episodes with model={args.model}")

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
    summary_path = args.output_dir / "openai_agent_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary")
    print("-------")
    print(json.dumps(summary, indent=2))
    print(f"\nFrames saved in: {summary['output_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
