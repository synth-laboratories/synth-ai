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
import tomllib

# from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft import (
#     test_crafter_react_agent_lm_synth as runner,
# )
from examples.finetuning.synth_qwen import (
    react_agent_lm as runner,
)
from synth_ai.config.base_url import (
    PROD_BASE_URL_DEFAULT,
    get_learning_v2_base_url,
)

# Force prod by default for this runner unless explicitly overridden
_force_prod = os.getenv("CRAFTER_FORCE_PROD", "1").lower() in ("1", "true", "yes", "on")
if _force_prod:
    # Sanitize implicit local/dev overrides
    os.environ.pop("SYNTH_LOCAL_BASE_URL", None)
    os.environ.pop("SYNTH_DEV_BASE_URL", None)
    # If caller hasn't explicitly set LEARNING_V2_BASE_URL, lock to prod default
    if "LEARNING_V2_BASE_URL" not in os.environ:
        os.environ["LEARNING_V2_BASE_URL"] = PROD_BASE_URL_DEFAULT

# Resolve base URL from shared config (honors LEARNING_V2_BASE_URL and sanitized overrides)
os.environ["SYNTH_BASE_URL"] = get_learning_v2_base_url()

print(f"🔧 Using Synth base URL = {os.environ.get('SYNTH_BASE_URL')}")

cfg_path = os.getenv("CRAFTER_CONFIG", "examples/finetuning/synth_qwen/config.toml")
cfg = {}
if os.path.exists(cfg_path):
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
else:
    cfg = {"rollouts": {}}
rcfg = cfg.get("rollouts", {})

MODEL_ID = os.getenv("CRAFTER_MODEL", rcfg.get("model", "Qwen/Qwen3-4B-Instruct-2507"))
EPISODES = os.getenv("CRAFTER_EPISODES", str(rcfg.get("episodes", 10)))
MAX_STEPS = os.getenv("CRAFTER_MAX_STEPS", str(rcfg.get("max_steps", 30)))
DIFFICULTY = os.getenv("CRAFTER_DIFFICULTY", rcfg.get("difficulty", "easy"))


async def main() -> None:
    think_env = os.getenv("CRAFTER_THINK", "0").lower()
    enable_think = think_env in ("1", "true", "yes", "on")
    think_flag = "--think" if enable_think else "--no-think"

    # Tighten prompts and enforce tool calling like the tests do
    os.environ["CRAFTER_STOP_AFTER_TOOL_CALLS"] = "1"
    os.environ["SYNTH_OPENAI_DEBUG"] = "0"
    os.environ["CRAFTER_MAX_TOKENS"] = os.environ.get(
        "CRAFTER_MAX_TOKENS", str(rcfg.get("max_tokens", 2048))
    )
    os.environ["CRAFTER_TOOL_CHOICE"] = os.environ.get(
        "CRAFTER_TOOL_CHOICE", rcfg.get("tool_choice", "required")
    )
    os.environ["CRAFTER_TEMPERATURE"] = os.environ.get(
        "CRAFTER_TEMPERATURE", str(rcfg.get("temperature", 0.4))
    )

    # Default v3 traces path from config if not already set
    tcfg = cfg.get("traces", {})
    if "SQLD_DB_PATH" not in os.environ and tcfg.get("sqld_db_path"):
        os.environ["SQLD_DB_PATH"] = tcfg["sqld_db_path"]
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
