#!/usr/bin/env python3
"""Minimal wrapper to run the modern Crafter-Classic evaluation (tracing_v3) on
Synth’s Qwen-7B model.

It forwards CLI flags to the canonical `test_crafter_react_agent_lm_synth.py`
implementation that already has full v3-tracing, warm-up, and reporting logic.
"""

import asyncio
import os
import sys
from pathlib import Path

# Bridge older env var name to new one so the runner uses the local backend
if "SYNTH_API_URL" in os.environ and "SYNTH_BASE_URL" not in os.environ:
    _base = os.environ["SYNTH_API_URL"].rstrip("/")
    if not _base.endswith("/api"):
        _base = f"{_base}/api"
    os.environ["SYNTH_BASE_URL"] = _base

print(f"🔧 Using Synth base URL = {os.environ.get('SYNTH_BASE_URL')}")

# ---------------------------------------------------------------------------
# Configuration (override with env vars if desired)
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("CRAFTER_MODEL", "Qwen/Qwen3-14B")
EPISODES = os.getenv("CRAFTER_EPISODES", "1")  # default 1 trajectory per run
MAX_STEPS = os.getenv("CRAFTER_MAX_STEPS", "30")
DIFFICULTY = os.getenv("CRAFTER_DIFFICULTY", "easy")  # easy|normal|hard

# ---------------------------------------------------------------------------
# Locate the reference runner within the package
# ---------------------------------------------------------------------------

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

# Import the runner‘s `main` entrypoint dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("crafter_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)  # type: ignore[var-annotated]
spec.loader.exec_module(runner)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Execute the runner with tailored argv
# ---------------------------------------------------------------------------

# No monkeypatch needed: the runner forwards extra_body to the LM.

async def main() -> None:
    # Prepare argv for the runner – mimics CLI usage
    think_env = os.getenv("CRAFTER_THINK", "0").lower()
    enable_think = think_env in ("1", "true", "yes", "on")
    think_flag = "--think" if enable_think else "--no-think"

    # Force stop-after-tool-calls and enable client debug in-code (no shell exports needed)
    os.environ["CRAFTER_STOP_AFTER_TOOL_CALLS"] = "1"
    os.environ["SYNTH_OPENAI_DEBUG"] = "0"

    # Set sampling params in code (no CLI globals)
    os.environ["CRAFTER_MAX_TOKENS"] = os.environ.get("CRAFTER_MAX_TOKENS", "2048")
    os.environ["CRAFTER_TOOL_CHOICE"] = os.environ.get("CRAFTER_TOOL_CHOICE", "required")
    os.environ["CRAFTER_TEMPERATURE"] = os.environ.get("CRAFTER_TEMPERATURE", "0.1")

    # Override the system prompt to strongly require a single tool call
    os.environ["CRAFTER_SYSTEM_PROMPT"] = (
        "You are CrafterAgent playing the Crafter survival environment. Your goal is to stay alive and unlock as many achievements as possible. "
        "Keep your reasoning brief and focus on the tool call."
        # "Do NOT include any natural language before or after the tool call. "
        "ALWAYS provide 2-5 actions."  # Use the interact tool and
        "Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, "
        "make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop."
    )
    # Suppress per-turn reminder text in observations to keep prompts tighter
    os.environ["CRAFTER_SUPPRESS_OBS_REMINDER"] = "1"
    os.environ["CRAFTER_EPISODES"] = "3"

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
        "--quiet",  # suppress extremely verbose output – remove if you want full logs
    ]

    await runner.main()  # runner.main is async


if __name__ == "__main__":
    asyncio.run(main())


# 🔥 Warming up Qwen/Qwen3-8B on Synth backend...
# ✅ Warmed Qwen/Qwen3-8B in 13s        meout elapsed=10s
# ✅ Model warmed up successfully!

# 🚀 Starting sqld daemon for v3 tracing...
# ✅ sqld daemon started

# 📊 V3 Tracing enabled. Traces will be saved to: ./traces_v3_lm_synth/traces.db
#    Experiment: crafter_lm_synth_Qwen/Qwen3-8B_20250807_170718

# 🚀 Running 3 episodes...

# 📤 Starting episodes...
# Episode 0:   7%|▊           | 2/30 [01:05<15:22, 32.95s/it, tc=1, act=2, tok=910, in=644, tps=13.82]🔍 DEBUG: consecutive_no_tool_calls=1                                                  | 0/30 [00:00<?, ?it/s]
# Episode 2:  10%|█▎           | 3/30 [01:20<11:35, 25.75s/it, tc=1, act=1, tok=386, in=647, tps=22.8]🔍 DEBUG: consecutive_no_tool_calls=1| 2/30 [01:33<21:50, 46.80s/it, tc=0, act=0, tok=2048, in=644, tps=21.89]
# Episode 0:  10%|█          | 3/30 [02:33<25:02, 55.64s/it, tc=0, act=0, tok=2048, in=646, tps=23.44]🔍 DEBUG: consecutive_no_tool_calls=1| 3/30 [02:43<25:19, 56.29s/it, tc=1, act=1, tok=1636, in=684, tps=23.54]
# Episode 0:  13%|█▍         | 4/30 [03:20<22:44, 52.49s/it, tc=1, act=4, tok=1092, in=686, tps=23.31]🔍 DEBUG: consecutive_no_tool_calls=1
# Episode 0:  20%|██▍         | 6/30 [04:42<17:54, 44.76s/it, tc=1, act=3, tok=633, in=646, tps=22.83]🔍 DEBUG: consecutive_no_tool_calls=1| 4/30 [04:12<29:40, 68.50s/it, tc=0, act=0, tok=2048, in=644, tps=23.03]
#                                                                                                     🔍 DEBUG: consecutive_no_tool_calls=2
# Episode 2:  20%|██▍         | 6/30 [04:50<23:20, 58.36s/it, tc=0, act=0, tok=2048, in=646, tps=23.4]🔍 DEBUG: consecutive_no_tool_calls=1| 5/30 [05:40<31:26, 75.45s/it, tc=0, act=0, tok=2048, in=684, tps=23.28]
# Episode 0:  23%|██▊         | 7/30 [06:11<22:32, 58.79s/it, tc=0, act=0, tok=2048, in=648, tps=23.2]

#  V3 Tracing enabled. Traces will be saved to: ./traces_v3_lm_synth/traces.db
#    Experiment: crafter_lm_synth_Qwen/Qwen3-14B_20250807_172630

# 🚀 Running 3 episodes...

# 📤 Starting episodes...
# Episode 0:  20%|██▏        | 6/30 [04:36<18:44, 46.85s/it, tc=1, act=3, tok=1219, in=646, tps=22.31]🔍 DEBUG: consecutive_no_tool_calls=1| 6/30 [04:55<22:17, 55.71s/it, tc=1, act=3, tok=1449, in=664, tps=21.82]
# Episode 2:  17%|█▊         | 5/30 [04:09<22:15, 53.44s/it, tc=1, act=3, tok=1109, in=650, tps=22.34]🔍 DEBUG: consecutive_no_tool_calls=1
# Episode 0:  23%|██▌        | 7/30 [06:08<23:29, 61.28s/it, tc=0, act=0, tok=2048, in=652, tps=22.34]c
# Episode 1:  23%|██▌        | 7/30 [06:24<25:25, 66.33s/it, tc=1, act=3, tok=1946, in=668, tps=21.97]
# Episode 2:  23%|██▌        | 7/30 [07:00<26:56, 70.26s/it, tc=1, act=6, tok=1769, in=692, tps=22.38]