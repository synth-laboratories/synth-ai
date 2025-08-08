#!/usr/bin/env python3
"""
Run Crafter-Classic evaluation (v3 tracing) on Synth’s Qwen 4B model.

This forwards flags into the canonical `test_crafter_react_agent_lm_synth.py`
runner that already handles v3 tracing, warm-up, and reporting.

Environment variables:
- CRAFTER_MODEL (default: Qwen/Qwen3-4B-Instruct-2507)
- CRAFTER_EPISODES (default: 10)
- CRAFTER_MAX_STEPS (default: 30)
- CRAFTER_DIFFICULTY (default: easy)
- CRAFTER_THINK (default: 0 -> use --no-think)

It also sets a few runner-specific env flags to enforce short outputs and a single tool call.
"""
import asyncio
import os
import sys

# from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft import (
#     test_crafter_react_agent_lm_synth as runner,
# )
from examples.finetuning.synth_qwen import (
    react_agent_lm as runner,
)
from synth_ai.config.base_url import get_learning_v2_base_url

# Ensure SYNTH_BASE_URL is set for the runner
if "SYNTH_BASE_URL" not in os.environ:
    # Bridge older env var name to new one first
    if "SYNTH_API_URL" in os.environ:
        _base = os.environ["SYNTH_API_URL"].rstrip("/")
        if not _base.endswith("/api"):
            _base = f"{_base}/api"
        os.environ["SYNTH_BASE_URL"] = _base
    else:
        # Fall back to standardized resolver (defaults to prod, honors .env overrides)
        os.environ["SYNTH_BASE_URL"] = get_learning_v2_base_url()

print(f"🔧 Using Synth base URL = {os.environ.get('SYNTH_BASE_URL')}")

MODEL_ID = os.getenv("CRAFTER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
EPISODES = os.getenv("CRAFTER_EPISODES", "10")
MAX_STEPS = os.getenv("CRAFTER_MAX_STEPS", "30")
DIFFICULTY = os.getenv("CRAFTER_DIFFICULTY", "easy")


async def main() -> None:
    think_env = os.getenv("CRAFTER_THINK", "0").lower()
    enable_think = think_env in ("1", "true", "yes", "on")
    think_flag = "--think" if enable_think else "--no-think"

    # Tighten prompts and enforce tool calling like the tests do
    os.environ["CRAFTER_STOP_AFTER_TOOL_CALLS"] = "1"
    os.environ["SYNTH_OPENAI_DEBUG"] = "0"
    os.environ["CRAFTER_MAX_TOKENS"] = os.environ.get("CRAFTER_MAX_TOKENS", "2048")
    os.environ["CRAFTER_TOOL_CHOICE"] = os.environ.get("CRAFTER_TOOL_CHOICE", "required")
    os.environ["CRAFTER_TEMPERATURE"] = os.environ.get("CRAFTER_TEMPERATURE", "0.4")
    os.environ["CRAFTER_SYSTEM_PROMPT"] = (
        "You are CrafterAgent playing the Crafter survival environment. Your goal is to stay alive and unlock as many achievements as possible. "
        "Keep your reasoning very brief and focus on the tool call. Use the tool available to you to play Crafter"
        "ALWAYS provide 2-5 actions. Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, "
        "make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop."
    )
    os.environ["CRAFTER_SUPPRESS_OBS_REMINDER"] = "1"
    # Ensure we log full LM inputs and tools
    os.environ["CRAFTER_LOG_FULL_INPUTS"] = os.environ.get("CRAFTER_LOG_FULL_INPUTS", "1")

    sys.argv = [
        "crafter_runner",
        "--model",
        MODEL_ID,
        "--episodes",
        str(EPISODES),
        "--max-steps",
        str(MAX_STEPS),
        "--difficulty",
        DIFFICULTY,
        think_flag,
        "--quiet",
    ]

    await runner.main()


if __name__ == "__main__":
    asyncio.run(main())