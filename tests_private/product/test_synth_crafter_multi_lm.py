#!/usr/bin/env python3
"""Run Crafter evaluation via LM against multiple providers/models (configurable).

Runs up to three variants sequentially:
- OpenAI: gpt-4.1-mini
- Groq:   qwen/qwen3-32b (override via GROQ_QWEN3_32B_MODEL)
- Groq:   openai/gpt-oss-20b (override via GROQ_GPT_OSS_20B_MODEL)

Control which to run with CRAFTER_MULTI_RUNS env (all|openai|groq-qwen|groq-gptoss, comma-separated allowed).

Environment variables expected:
- OPENAI_API_KEY
- GROQ_API_KEY
- Optional overrides: GROQ_BASE_URL (default https://api.groq.com/openai/v1)
- Common Crafter vars: CRAFTER_EPISODES, CRAFTER_MAX_STEPS, CRAFTER_DIFFICULTY, CRAFTER_THINK
"""

import asyncio
import os
import sys
from pathlib import Path


# Locate the canonical runner from the repo root (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "synth_ai"
    / "environments"
    / "examples"
    / "crafter_classic"
    / "agent_demos"
    / "crafter_modal_ft"
    / "test_crafter_react_agent_lm_synth.py"
)
if not RUNNER_PATH.exists():
    raise FileNotFoundError(f"Crafter runner not found at {RUNNER_PATH}")

import importlib.util
spec = importlib.util.spec_from_file_location("crafter_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)  # type: ignore[var-annotated]
spec.loader.exec_module(runner)  # type: ignore[union-attr]


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.lower()
    return v in ("1", "true", "yes", "on")


async def run_variant(display: str, base_url: str, api_key_env: str, model_id: str, episodes: str, max_steps: str, difficulty: str, think: bool) -> None:
    print(f"\n✨ Running: {display}")

    # Configure OpenAI-compatible endpoint for the runner via Synth-compatible envs
    os.environ["SYNTH_BASE_URL"] = base_url.rstrip("/")
    os.environ["SYNTH_API_KEY"] = os.getenv(api_key_env, "")
    if not os.environ["SYNTH_API_KEY"]:
        raise RuntimeError(f"Missing API key for {display}: expected {api_key_env}")

    # Tight system prompt and observation suppression
    os.environ["CRAFTER_SYSTEM_PROMPT"] = (
        "You are CrafterAgent playing the Crafter survival environment. Your goal is to stay alive and unlock as many achievements as possible. "
        "Keep your reasoning brief and focus on the tool call. ALWAYS provide 2-5 actions. "
        "Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, "
        "make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop."
    )
    os.environ["CRAFTER_SUPPRESS_OBS_REMINDER"] = "1"

    # Thinking flag
    think_flag = "--think" if think else "--no-think"

    # Skip warmup for external providers
    skip_warmup_flag = "--skip-warmup"

    sys.argv = [
        "crafter_runner",
        "--model", model_id,
        "--max-tokens", "2048",
        "--tool-choice", "required",
        "--temperature", "0.1",
        "--episodes", episodes,
        "--max-steps", max_steps,
        "--difficulty", difficulty,
        skip_warmup_flag,
        think_flag,
        "--quiet",
    ]

    await runner.main()


async def main() -> None:
    # Common controls
    episodes = os.getenv("CRAFTER_EPISODES", "1")
    max_steps = os.getenv("CRAFTER_MAX_STEPS", "50")
    difficulty = os.getenv("CRAFTER_DIFFICULTY", "easy")
    think = _bool_env("CRAFTER_THINK", False)

    # Selection of which variants to run
    selection = os.getenv("CRAFTER_MULTI_RUNS", "all").lower()
    selected = set(x.strip() for x in selection.split(","))
    if "all" in selected:
        selected = {"groq-qwen"}  # "openai", "groq-qwen",

    # Provider endpoints and models
    openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    qwen3_model = os.getenv("GROQ_QWEN3_32B_MODEL", "Qwen/Qwen3-32B")
    gptoss_model = os.getenv("GROQ_GPT_OSS_20B_MODEL", "openai/gpt-oss-120b")

    # Execute selected runs sequentially
    if "openai" in selected:
        await run_variant(
            display="OpenAI gpt-4.1-mini",
            base_url=openai_base,
            api_key_env="OPENAI_API_KEY",
            model_id="gpt-4.1-mini",
            episodes=episodes,
            max_steps=max_steps,
            difficulty=difficulty,
            think=think,
        )

    if "groq-qwen" in selected:
        await run_variant(
            display="Groq Qwen/Qwen3-32B",
            base_url=groq_base,
            api_key_env="GROQ_API_KEY",
            model_id=qwen3_model,
            episodes=episodes,
            max_steps=max_steps,
            difficulty=difficulty,
            think=think,
        )

    if "groq-gptoss" in selected:
        await run_variant(
            display="Groq openai/gpt-oss-120b",
            base_url=groq_base,
            api_key_env="GROQ_API_KEY",
            model_id=gptoss_model,
            episodes=episodes,
            max_steps=max_steps,
            difficulty=difficulty,
            think=think,
        )


if __name__ == "__main__":
    asyncio.run(main())

