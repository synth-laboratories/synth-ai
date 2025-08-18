#!/usr/bin/env python3
"""
Filter v3 Crafter traces into an SFT-ready JSONL using the maintained
Modal/Synth filter logic (no CLI needed). Intended to be run after
collecting trajectories with the Crafter runner.

Environment:
- CRAFTER_DB_URL (default: sqlite:///traces_v3_lm_synth/traces.db)
- OUTPUT_JSONL (default: ft_data/qwen4b_crafter_sft.jsonl)
- MIN_TOTAL_REWARD (float, default: 1.0)
- MIN_ACHIEVEMENTS (int, default: 0)
- MAX_COST (float, default: 10.0)
- MAX_TOKENS (int, default: 100000)
- MODELS (optional, space-separated model names; default empty = all)
"""

import asyncio
import json
import os
from typing import Any

# Reuse the existing filtering implementation
from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso import (
    filter_traces_from_turso,
)


def build_config() -> dict[str, Any]:
    models_env = os.getenv("MODELS", "").strip()
    models: list[str] = models_env.split() if models_env else []
    return {
        "mode": "trajectory",
        "filters": {
            "min_total_reward": float(os.getenv("MIN_TOTAL_REWARD", "1.0")),
            "min_achievements": int(os.getenv("MIN_ACHIEVEMENTS", "0")),
            "max_cost": float(os.getenv("MAX_COST", "10.0")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "100000")),
            "models": models,
        },
    }


async def main() -> None:
    db_url = os.getenv("CRAFTER_DB_URL", "sqlite:///traces_v3_lm_synth/traces.db")
    output_path = os.getenv("OUTPUT_JSONL", "ft_data/qwen4b_crafter_sft.jsonl")
    config = build_config()

    print("🤖 Modal/Synth Fine-Tuning Data Filter (v3)")
    print("Using database:", db_url)
    print("Output file:", output_path)
    print("Config:", json.dumps(config, indent=2))

    num_examples, stats = await filter_traces_from_turso(db_url, output_path, config)

    print("\n✅ Wrote", num_examples, "training examples to", output_path)
    print("📊 Stats keys:", list(stats.keys()))


if __name__ == "__main__":
    asyncio.run(main())
