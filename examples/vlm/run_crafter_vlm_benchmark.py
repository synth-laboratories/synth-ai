#!/usr/bin/env python3
"""
Benchmark Crafter performance across prompt modalities (text-only, image-only, both).

For each mode we:
  * Run 20 seeded episodes (configurable) with GPT-4o mini via OpenAI Chat Completions.
  * Execute the returned tool calls in the local Crafter environment.
  * Record achievements/steps and save every rendered frame under `examples/vlm/temp/`.

Concurrency is capped by an asyncio semaphore (default parallelism = 10).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.warming_up_to_rl.task_app.synth_envs_hosted.envs.crafter.environment import (
    CrafterEnvironmentWrapper,
)
from examples.warming_up_to_rl.task_app.synth_envs_hosted.envs.crafter.policy import CrafterPolicy
from openai import AsyncOpenAI
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

OUTPUT_ROOT = Path("examples/vlm/temp")


class Mode(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    BOTH = "both"


@dataclass
class EpisodeResult:
    mode: Mode
    seed: int
    steps_taken: int
    achievements: set[str]
    total_reward: float
    tool_calls: int


def _ensure_openai_client(api_key: str | None) -> AsyncOpenAI:
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set to run the VLM benchmark (export the key or add to your .env)."
        )
    return AsyncOpenAI(api_key=api_key)


def _build_task_instance(seed: int) -> CrafterTaskInstance:
    impetus = Impetus(instructions="Explore, survive, and unlock achievements.")
    intent = Intent(rubric={"goal": "Unlock achievements"}, gold_trajectories=None, gold_state_diff={})
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
    # Engine expects these config keys
    instance.config = {"seed": seed, "length": 256, "area": [64, 64]}
    return instance


def _save_observation_frame(observation_packet: dict[str, Any], dest_path: Path) -> None:
    obs = observation_packet.get("observation")
    if not isinstance(obs, dict):
        return
    image_b64 = obs.get("observation_image_base64")
    if not isinstance(image_b64, str) or not image_b64:
        return
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(base64.b64decode(image_b64))
    except Exception:
        pass  # best effort


def _strip_image_fields(observation_packet: dict[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(observation_packet))
    obs = stripped.get("observation")
    if isinstance(obs, dict):
        for key in list(obs.keys()):
            if key.startswith("observation_image"):
                obs.pop(key, None)
    return stripped


def _make_image_only_request(request: dict[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(request))
    for message in cloned.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            image_parts = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") in {"image_url", "image"}
            ]
            message["content"] = image_parts or content
        elif isinstance(content, str):
            # No structured parts available; leave as empty string
            message["content"] = ""
    return cloned


async def _run_episode(
    *,
    mode: Mode,
    seed: int,
    client: AsyncOpenAI,
    model: str,
    max_steps: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> EpisodeResult:
    async with semaphore:
        task_instance = _build_task_instance(seed)
        env = CrafterClassicEnvironment(task_instance)
        wrapper = CrafterEnvironmentWrapper(env, seed=seed)

        policy = CrafterPolicy(inference_url="openai://chat-completions", model=model)
        await policy.initialize({"use_tools": True, "model": model})

        observation_packet = await wrapper.initialize()
        achievements: set[str] = set()
        total_reward = 0.0
        steps_taken = 0
        tool_calls_total = 0

        frames_dir = OUTPUT_ROOT / f"{mode.value}_frames" / f"seed_{seed:04d}"
        _save_observation_frame(observation_packet, frames_dir / "step_000.png")

        for step_idx in range(max_steps):
            obs_dict = observation_packet.get("observation")
            if not isinstance(obs_dict, dict):
                break

            observation_for_policy: dict[str, Any]
            metadata_payload: dict[str, Any] = {}

            if mode == Mode.TEXT:
                observation_for_policy = _strip_image_fields(observation_packet)
            else:
                observation_for_policy = json.loads(json.dumps(observation_packet))
                metadata_payload["raw_observation"] = observation_packet

            obs_text = policy._format_observation_for_llm(observation_for_policy)  # noqa: SLF001
            _, meta = await policy.step(
                observation_text=obs_text,
                metadata=metadata_payload,
            )
            inference_request = json.loads(json.dumps(meta["inference_request"]))

            if mode == Mode.IMAGE:
                inference_request = _make_image_only_request(inference_request)

            inference_request.update(
                {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": inference_request.get("max_tokens", 512),
                }
            )
            inference_request.pop("stop_after_tool_calls", None)
            inference_request.pop("thinking_mode", None)
            inference_request.pop("thinking_budget", None)

            response = await client.chat.completions.create(**inference_request)
            response_dict = response.model_dump()

            assistant_tool_calls = CrafterPolicy.parse_response_to_tool_calls(
                response_dict,
                use_tools=policy.use_tools,
            )
            if not assistant_tool_calls:
                break

            tool_calls_total += len(assistant_tool_calls)
            assistant_message = response_dict["choices"][0].get("message") or {}
            assistant_text = assistant_message.get("content")

            env_response = await wrapper.step(assistant_tool_calls)
            if not isinstance(env_response, dict):
                raise RuntimeError(f"Unexpected environment response type: {type(env_response)!r}")

            policy._append_assistant_turn(  # noqa: SLF001
                assistant_text,
                assistant_tool_calls,
                env_response,
            )

            steps_taken += 1
            obs = env_response.get("observation")
            if isinstance(obs, dict):
                ach = obs.get("achievements_status")
                if isinstance(ach, dict):
                    for name, unlocked in ach.items():
                        if unlocked:
                            achievements.add(str(name))
                reward = obs.get("reward_last_step")
                if isinstance(reward, (int, float)):
                    total_reward += float(reward)

            _save_observation_frame(env_response, frames_dir / f"step_{step_idx + 1:03d}.png")

            if env_response.get("done"):
                break
            observation_packet = env_response

        await wrapper.terminate()
        return EpisodeResult(
            mode=mode,
            seed=seed,
            steps_taken=steps_taken,
            achievements=achievements,
            total_reward=total_reward,
            tool_calls=tool_calls_total,
        )


def _summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    grouped: dict[Mode, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)

    summary: dict[str, Any] = {}
    for mode, mode_results in grouped.items():
        if not mode_results:
            continue
        mean_steps = sum(r.steps_taken for r in mode_results) / len(mode_results)
        mean_achievements = sum(len(r.achievements) for r in mode_results) / len(mode_results)
        achievement_counts = Counter()
        for res in mode_results:
            achievement_counts.update(res.achievements)
        summary[mode.value] = {
            "episodes": len(mode_results),
            "mean_steps": round(mean_steps, 2),
            "mean_achievements": round(mean_achievements, 2),
            "total_tool_calls": sum(r.tool_calls for r in mode_results),
            "achievements": {name: count for name, count in sorted(achievement_counts.items())},
        }
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="OpenAI model id to benchmark")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per mode")
    parser.add_argument("--steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent OpenAI calls")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    client = _ensure_openai_client(api_key)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks: list[asyncio.Task[EpisodeResult]] = []
    for mode in (Mode.TEXT, Mode.IMAGE, Mode.BOTH):
        for seed in range(args.seeds):
            task = asyncio.create_task(
                _run_episode(
                    mode=mode,
                    seed=seed,
                    client=client,
                    model=args.model,
                    max_steps=args.steps,
                    temperature=args.temperature,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    summary = _summarise(results)

    summary_path = OUTPUT_ROOT / "vlm_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBenchmark Summary")
    print("-----------------")
    print(json.dumps(summary, indent=2))
    print(f"\nFrames stored under: {OUTPUT_ROOT}/<mode>_frames/seed_xxxx/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
