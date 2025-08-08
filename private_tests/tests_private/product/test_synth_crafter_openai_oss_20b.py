#!/usr/bin/env python3
"""Minimal wrapper to run the modern Crafter-Classic evaluation (tracing_v3) on
Synth’s GPT-OSS-20B model.

It reuses the same heavy-duty runner that already implements tracing_v3, warm-up
and reporting – we just feed in the model name and episode count.
"""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# Bridge older env var to the one used by the OpenAI client, and ensure /api suffix
if "SYNTH_API_URL" in os.environ and "SYNTH_BASE_URL" not in os.environ:
    _base = os.environ["SYNTH_API_URL"].rstrip("/")
    if not _base.endswith("/api"):
        _base = f"{_base}/api"
    os.environ["SYNTH_BASE_URL"] = _base

print(f"🔧 Using Synth base URL = {os.environ.get('SYNTH_BASE_URL')}")

MODEL_ID = os.getenv("CRAFTER_MODEL", "openai/gpt-oss-20b")
EPISODES = os.getenv("CRAFTER_EPISODES", "1")
MAX_STEPS = os.getenv("CRAFTER_MAX_STEPS", "30")
DIFFICULTY = os.getenv("CRAFTER_DIFFICULTY", "easy")

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

spec = importlib.util.spec_from_file_location("crafter_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)  # type: ignore[var-annotated]
spec.loader.exec_module(runner)  # type: ignore[union-attr]


async def main() -> None:
    # Force the prompt variant that worked best with required + temp=1
    os.environ["CRAFTER_SYSTEM_PROMPT"] = (
        "You are CrafterAgent playing the Crafter survival environment. Your goal is to stay alive and unlock as many achievements as possible. "
        "Use the 7x7 semantic map view to navigate toward resources and interact with nearby objects. "
        #"Reasoning: medium"
        "Available actions: move_left, move_right, move_up, move_down, do, sleep, "
        "place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, "
        "make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop. "
        # "Output only a single tool call."
    )
    # Suppress the observation-level reminder/example lines
    os.environ["CRAFTER_SUPPRESS_OBS_REMINDER"] = "1"

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
        "--temperature",
        "1",
        "--max-tokens",
        "4096",
        "--tool-choice",
        "required",
        "--quiet",
    ]

    await runner.main()


if __name__ == "__main__":
    asyncio.run(main())
