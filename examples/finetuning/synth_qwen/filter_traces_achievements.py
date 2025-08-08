#!/usr/bin/env python3
"""
Filter v3 Crafter traces into SFT JSONL requiring specific achievements.

Environment:
- CRAFTER_DB_URL (default: sqlite:///traces_v3_lm_synth/traces.db)
- OUTPUT_JSONL (default: ft_data/qwen4b_crafter_sft_ach.jsonl)
- REQUIRED_ACHIEVEMENTS (space-separated, default: collect_wood)
- MIN_TOTAL_REWARD (float, default: 0.0)
- MAX_COST (float, default: inf)
- MAX_TOKENS (int, default: inf)
- MODELS (optional, space-separated model names; default empty = all)
"""
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso import (
    FinetuningDataExtractorV3,
)


def env_list(name: str) -> List[str]:
    val = os.getenv(name, "").strip()
    return val.split() if val else []


def normalize_db_url(raw: str) -> str:
    # Accept file path or sqlite URLs; ensure async driver prefix
    if raw.endswith(".db") and not raw.startswith("sqlite"):
        return f"sqlite+aiosqlite:///{raw}"
    if raw.startswith("sqlite+aiosqlite:///"):
        return raw
    if raw.startswith("sqlite:///") and raw.endswith(".db"):
        return raw.replace("sqlite:///", "sqlite+aiosqlite:///")
    return raw


def build_filters() -> Dict[str, Any]:
    req = set(env_list("REQUIRED_ACHIEVEMENTS") or ["collect_wood"])
    models = env_list("MODELS")
    min_reward = float(os.getenv("MIN_TOTAL_REWARD", "0.0"))
    max_cost = float(os.getenv("MAX_COST", str(math.inf)))
    max_tokens = int(os.getenv("MAX_TOKENS", str(1_000_000_000)))
    return {
        "required_achievements": req,
        "models": models,
        "min_total_reward": min_reward,
        "max_cost": max_cost,
        "max_tokens": max_tokens,
    }


async def main() -> None:
    raw_db_url = os.getenv("CRAFTER_DB_URL", "sqlite:///traces_v3_lm_synth/traces.db")
    db_url = normalize_db_url(raw_db_url)
    output_path = os.getenv("OUTPUT_JSONL", "ft_data/qwen4b_crafter_sft_ach.jsonl")
    filters = build_filters()

    print("🤖 Modal/Synth FT Filter (achievements)")
    print("Using database:", db_url)
    print("Output file:", output_path)
    print("Filters:", json.dumps({k: (list(v) if isinstance(v, set) else v) for k, v in filters.items()}, indent=2))

    required: Set[str] = filters["required_achievements"]
    models: List[str] = filters["models"]
    min_reward: float = filters["min_total_reward"]
    max_cost: float = filters["max_cost"]
    max_tokens: int = filters["max_tokens"]

    stats: Dict[str, Any] = {
        "total_sessions": 0,
        "kept_sessions": 0,
        "total_examples": 0,
    }

    async with FinetuningDataExtractorV3(db_url) as extractor:
        all_sessions = await extractor.get_all_sessions()
        stats["total_sessions"] = len(all_sessions)

        kept: List[str] = []
        for _, row in all_sessions.iterrows():
            session_id = row["session_id"]
            metrics = await extractor.get_session_metrics(session_id)

            if metrics["total_reward"] < min_reward:
                continue
            if metrics["total_cost"] > max_cost:
                continue
            if metrics["total_tokens"] > max_tokens:
                continue

            if models:
                model_query = (
                    """
                    SELECT DISTINCT model_name
                    FROM events
                    WHERE session_id = :session_id
                      AND event_type = 'cais'
                      AND model_name IS NOT NULL
                    """
                )
                model_df = await extractor.db_manager.query_traces(model_query, {"session_id": session_id})
                session_models = model_df["model_name"].tolist() if not model_df.empty else []
                if not any(m in models for m in session_models):
                    continue

            achievements = await extractor.get_session_achievements(session_id)
            if not achievements:
                continue
            if not (required & set(achievements)):
                continue

            kept.append(session_id)

        stats["kept_sessions"] = len(kept)

        training_data = await extractor.extract_openai_format(kept)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in training_data:
                f.write(json.dumps(ex) + "\n")
        stats["total_examples"] = len(training_data)

    print("\n✅ Wrote", stats["total_examples"], "examples from", stats["kept_sessions"], "sessions")


if __name__ == "__main__":
    asyncio.run(main())