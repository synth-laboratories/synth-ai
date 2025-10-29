#!/usr/bin/env python3
"""
Collect Crafter vision traces for SFT dataset creation.

Supports both:
1. OpenAI models (gpt-5-nano, gpt-4o-mini) via OpenAI API
2. Qwen-VL models via synth-ai hosted inference

Traces are stored in SQLite with full multimodal messages (text + base64 images)
ready for export to SFT JSONL format.

Requirements:
  - For OpenAI: OPENAI_API_KEY environment variable
  - For synth-ai: SYNTH_API_KEY environment variable

Usage:
  # Collect with gpt-5-nano
  uv run python examples/qwen_vl/collect_vision_traces.py \
      --model gpt-5-nano \
      --provider openai \
      --episodes 100 \
      --max-steps 50 \
      --output-dir traces/gpt5nano_vision

  # Collect with Qwen3-VL via synth
  uv run python examples/qwen_vl/collect_vision_traces.py \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --provider synth \
      --episodes 100 \
      --max-steps 50 \
      --output-dir traces/qwen3vl_vision
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast
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

# Try importing trace storage
try:
    from synth_ai.tracing_v3.storage import create_storage
    from synth_ai.tracing_v3.storage.config import StorageBackend, StorageConfig
    TRACING_AVAILABLE = True
except ImportError:
    print("Warning: Tracing storage not available. Traces will not be persisted.")
    TRACING_AVAILABLE = False


def _get_openai_client():
    """Get OpenAI client."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _default_backend_base_url() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    return raw if raw.endswith("/api") else f"{raw}/api"


def _get_synth_client():
    """Get synth-ai inference client."""
    from synth_ai.inference.client import InferenceClient
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY not set")
    base_url = os.getenv("SYNTH_BASE_URL", _default_backend_base_url())
    return InferenceClient(base_url=base_url, api_key=api_key)


def _build_task_instance(seed: int) -> CrafterTaskInstance:
    """Create Crafter task instance."""
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


def _normalise_openai_request(payload: dict[str, Any], model: str, temperature: float) -> dict[str, Any]:
    """Normalize inference request for OpenAI API."""
    request = dict(payload)
    request["model"] = model
    
    # Remove vendor-specific knobs
    request.pop("stop_after_tool_calls", None)
    request.pop("thinking_mode", None)
    request.pop("thinking_budget", None)
    
    # gpt-5 models have specific requirements
    if "gpt-5" in model.lower():
        # gpt-5-nano only supports temperature=1 (default)
        request.pop("temperature", None)  # Remove custom temperature
        request.setdefault("max_completion_tokens", 512)
        request.pop("max_tokens", None)  # Remove if present
    else:
        # Older models use max_tokens and support custom temperature
        request.setdefault("temperature", temperature)
        max_completion = request.pop("max_completion_tokens", None)
        if max_completion is not None:
            request["max_tokens"] = max_completion
        else:
            request.setdefault("max_tokens", 512)
    
    return request


async def collect_traces(
    model: str,
    provider: str,
    num_episodes: int,
    max_steps: int,
    seed_start: int,
    output_dir: Path,
    temperature: float,
):
    """Collect vision traces for SFT."""
    # Setup tracing store
    if not TRACING_AVAILABLE:
        raise RuntimeError("Tracing storage not available. Cannot persist traces.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "rollouts.db"
    storage_config = StorageConfig(
        backend=StorageBackend.SQLITE,
        connection_string=f"sqlite+aiosqlite:///{db_path}",
    )
    tracing_store = create_storage(storage_config)
    await tracing_store.initialize()
    
    # Setup inference client
    if provider == "openai":
        client = _get_openai_client()
        inference_url = "openai://chat-completions"
    elif provider == "synth":
        client = _get_synth_client()
        inference_url = "synth://inference"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"🎮 Collecting {num_episodes} episodes with {model}")
    print(f"   Provider: {provider}")
    print(f"   Max steps: {max_steps}")
    print(f"   Output: {output_dir}")
    print(f"   Database: {db_path}")
    print()
    
    total_steps = 0
    total_achievements = 0
    
    for episode_id in range(num_episodes):
        seed = seed_start + episode_id
        
        # Build task instance
        task_instance = _build_task_instance(seed)
        env = CrafterClassicEnvironment(task_instance)
        wrapper = CrafterEnvironmentWrapper(env, seed=seed)
        
        # Initialize policy (vision auto-detected from model name)
        policy = CrafterPolicy(inference_url=inference_url, model=model)
        await policy.initialize({
            "use_tools": True,
            "model": model,
            "temperature": temperature,
            "max_tokens": 512,
        })
        
        observation_packet = await wrapper.initialize()
        
        steps_taken = 0
        achievements = set()
        
        # Run episode
        for step_idx in range(max_steps):
            obs_dict = observation_packet.get("observation")
            if not isinstance(obs_dict, dict):
                break
            
            # Format observation
            obs_text = policy._format_observation_for_llm(observation_packet)  # noqa: SLF001
            
            # Get tool calls from policy
            tool_calls, meta = await policy.step(
                observation_text=obs_text,
                metadata={"raw_observation": observation_packet},
            )
            if "inference_request" not in meta:
                break
            
            inference_request = meta["inference_request"]
            
            # Call inference
            if provider == "openai":
                normalized_request = _normalise_openai_request(
                    inference_request,
                    model=model,
                    temperature=temperature,
                )
                response = client.chat.completions.create(**normalized_request)
                response_dict = response.model_dump()
            else:  # synth
                response_dict = await client.create_chat_completion(
                    model=model,
                    messages=inference_request["messages"],
                    temperature=temperature,
                    max_tokens=512,
                    tools=inference_request.get("tools"),
                )
            
            # Parse tool calls
            assistant_tool_calls = CrafterPolicy.parse_response_to_tool_calls(
                response_dict,
                use_tools=policy.use_tools,
            )
            if not assistant_tool_calls:
                break
            
            # Store trace
            assistant_message = response_dict["choices"][0].get("message", {})
            trace_messages = inference_request["messages"] + [assistant_message]
            
            tracing_store_any = cast(Any, tracing_store)
            if hasattr(tracing_store_any, "store_trace"):
                await tracing_store_any.store_trace(
                    session_id=f"ep{episode_id:04d}",
                    step=step_idx,
                    messages=trace_messages,
                    model=model,
                    metadata={
                        "seed": seed,
                        "has_image": policy.use_vision,
                        "provider": provider,
                    },
                )
            else:
                logging.warning(
                    "Tracing backend does not expose store_trace(); skipping persistence for episode %s",
                    episode_id,
                )
            
            # Execute action
            assistant_text = assistant_message.get("content")
            env_response = await wrapper.step(assistant_tool_calls)
            if not isinstance(env_response, dict):
                break
            
            # Update policy history
            policy._append_assistant_turn(  # noqa: SLF001
                assistant_text,
                assistant_tool_calls,
                env_response,
            )
            
            steps_taken += 1
            
            # Track achievements
            obs = env_response.get("observation", {})
            ach_status = obs.get("achievements_status", {})
            for name, unlocked in ach_status.items():
                if unlocked:
                    achievements.add(name)
            
            if env_response.get("done"):
                break
            observation_packet = env_response
        
        await wrapper.terminate()
        
        total_steps += steps_taken
        total_achievements += len(achievements)
        
        print(
            f"✓ Episode {episode_id:3d} (seed={seed}): {steps_taken} steps, "
            f"{len(achievements)} achievements"
        )
    
    print()
    print(f"✅ Collection complete!")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Total steps: {total_steps}")
    print(f"   Avg achievements: {total_achievements / num_episodes:.2f}")
    print(f"   Database: {db_path}")
    print()
    print("Next steps:")
    print("  1. Export traces to SFT JSONL format")
    print("  2. Split into train/val datasets")
    print("  3. Train VLM with LoRA")
    
    return db_path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., gpt-5-nano, Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "synth"],
        required=True,
        help="Inference provider",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces/vision_traces"),
        help="Output directory for traces",
    )
    args = parser.parse_args()
    
    await collect_traces(
        model=args.model,
        provider=args.provider,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    asyncio.run(main())
