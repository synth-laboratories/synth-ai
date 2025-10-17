#!/usr/bin/env python3
"""
Run a minimal Crafter agent that emits image-only prompts and saves rendered frames.

This script demonstrates the multimodal observation pipeline by:
  1. Initialising a `CrafterClassicEnvironment` with a deterministic seed.
  2. Capturing `observation_image_base64` at each step and writing PNG frames.
  3. Building OpenAI-style user messages that contain only an image part.
  4. Emitting a small JSONL preview of the messages so they can be inspected or fed
     directly into the fine-tuning dataset builder.

Usage:
    uv run python examples/vlm/crafter_image_only_agent.py --seed 7 --steps 5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

ACTION_NAME_TO_ID = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}


def _build_task_instance(seed: int) -> CrafterTaskInstance:
    """Construct a minimal Crafter task instance with the requested seed."""

    impetus = Impetus(instructions="Explore the world and survive.")
    intent = Intent(
        rubric={"goal": "Unlock achievements and stay alive."},
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
    # Attach environment config expected by the engine
    instance.config = {"seed": seed, "length": 256, "area": [64, 64]}
    return instance


def _select_actions(action_names: Iterable[str], steps: int) -> list[int]:
    resolved: list[int] = []
    names = list(action_names)
    if not names:
        names = ["move_right", "move_down", "move_left", "move_up", "do"]
    for idx in range(steps):
        name = names[idx % len(names)]
        action_id = ACTION_NAME_TO_ID.get(name)
        if action_id is None:
            raise ValueError(f"Unknown Crafter action: {name}")
        resolved.append(action_id)
    return resolved


def _save_base64_png(data: str, path: Path) -> None:
    """Decode a base64 string (with or without data URL prefix) and write to disk."""

    if data.startswith("data:"):
        _, _, encoded = data.partition(",")
    else:
        encoded = data
    path.write_bytes(base64.b64decode(encoded))


def _build_image_only_message(data_url: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": data_url}}],
    }


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    messages_path = output_dir / "image_only_messages.jsonl"

    task_instance = _build_task_instance(args.seed)
    env = CrafterClassicEnvironment(task_instance)

    # Initialise environment
    raw_obs = await env.initialize()
    observation = getattr(raw_obs, "observation", raw_obs)

    action_ids = _select_actions(args.actions, args.steps)
    records: list[dict[str, Any]] = []

    for step_idx in range(args.steps):
        obs_dict = observation if isinstance(observation, dict) else {}
        image_b64 = obs_dict.get("observation_image_base64")
        data_url = obs_dict.get("observation_image_data_url")

        if image_b64:
            frame_path = frames_dir / f"step_{step_idx:03d}.png"
            _save_base64_png(image_b64, frame_path)

        if data_url:
            message = _build_image_only_message(data_url)
        else:
            message = {
                "role": "user",
                "content": [{"type": "text", "text": "Image missing from observation."}],
            }

        records.append(
            {
                "step": step_idx,
                "action_id": action_ids[step_idx],
                "message": message,
                "observation_keys": sorted(obs_dict.keys()),
            }
        )

        # For the very first step, show the message structure
        if step_idx == 0:
            print("=== Image-only message example ===")
            print(json.dumps(message, indent=2))

        tool_call = EnvToolCall(tool="interact", args={"action": int(action_ids[step_idx])})
        env_step = await env.step(tool_call)
        observation = getattr(env_step, "observation", env_step)

    # Wrap up and dump the preview JSONL
    await env.terminate()
    with messages_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} frames -> {frames_dir}")
    print(f"Saved image-only message preview -> {messages_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Crafter environment seed")
    parser.add_argument("--steps", type=int, default=5, help="Number of env steps to capture")
    parser.add_argument(
        "--actions",
        nargs="*",
        default=["move_right", "move_down", "move_left", "move_up", "do"],
        help="Sequence of Crafter action names to cycle through",
    )
    default_output = Path("examples/vlm/temp")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for frames and message preview (default: {default_output})",
    )
    parser.add_argument(
        "--randomise",
        action="store_true",
        help="Shuffle the provided action sequence before running",
    )
    args = parser.parse_args()
    if args.randomise:
        random.shuffle(args.actions)
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
