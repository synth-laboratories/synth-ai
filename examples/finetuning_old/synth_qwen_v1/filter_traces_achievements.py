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
- WINDOW_MODE=1 to emit per-turn user→assistant examples
"""

import asyncio
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any

try:
    # Preferred path (modal-specific)
    from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso import (  # type: ignore
        FinetuningDataExtractorV3,
    )
except Exception:  # pragma: no cover
    try:
        # Fallback path used in some dist builds
        from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_openai_ft.filter_traces_sft_turso import (  # type: ignore
            FinetuningDataExtractorV3,
        )
    except Exception as _import_err:  # pragma: no cover
        raise ImportError(
            "Could not import FinetuningDataExtractorV3 from synth_ai."
        ) from _import_err


def env_list(name: str) -> list[str]:
    val = os.getenv(name, "").strip()
    return val.split() if val else []


def normalize_db_url(raw: str) -> str:
    # Accept file path or sqlite URLs; ensure async driver prefix
    if raw.endswith(".db") and not raw.startswith("sqlite"):
        return f"sqlite+aiosqlite:///{raw}"
    if raw.startswith("sqlite+aiosqlite///"):
        return raw
    if raw.startswith("sqlite///") and raw.endswith(".db"):
        return raw.replace("sqlite///", "sqlite+aiosqlite///")
    return raw


def build_filters() -> dict[str, Any]:
    cfg_default = Path(__file__).with_name("config.toml")
    cfg_path = os.getenv("CRAFTER_CONFIG", str(cfg_default))
    cfg: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    fcfg = cfg.get("filter", {})
    # Default: no required achievements gating unless provided via env/config
    req = set(env_list("REQUIRED_ACHIEVEMENTS") or fcfg.get("required_achievements", []))
    models = env_list("MODELS")
    # Default: allow zero reward unless overridden
    min_reward = float(os.getenv("MIN_TOTAL_REWARD", str(fcfg.get("min_total_reward", 0.0))))
    max_cost = float(os.getenv("MAX_COST", str(fcfg.get("max_cost", math.inf))))
    max_tokens = int(os.getenv("MAX_TOKENS", str(fcfg.get("max_tokens", 1_000_000_000))))
    return {
        "required_achievements": req,
        "models": models,
        "min_total_reward": min_reward,
        "max_cost": max_cost,
        "max_tokens": max_tokens,
    }


async def main() -> None:
    cfg_default = Path(__file__).with_name("config.toml")
    cfg_path = os.getenv("CRAFTER_CONFIG", str(cfg_default))
    cfg: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    fcfg = cfg.get("filter", {})
    tcfg = cfg.get("traces", {})

    # Prefer env; else derive from config or repo-local v3 path
    raw_db_url = os.getenv("CRAFTER_DB_URL", "")
    if not raw_db_url:
        db_path = fcfg.get("db_path")
        if not db_path and tcfg.get("sqld_db_path"):
            # derive the internal data file path from the sqld dir
            db_path = str(Path(tcfg["sqld_db_path"]) / "dbs" / "default" / "data")
        if db_path:
            raw_db_url = f"sqlite+aiosqlite:///{db_path}"
        else:
            # Try repo-local default: traces/v3/synth_ai.db/dbs/default/data
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / "traces" / "v3" / "synth_ai.db" / "dbs" / "default" / "data"
            raw_db_url = f"sqlite+aiosqlite:///{candidate}"
    db_url = normalize_db_url(raw_db_url)
    output_path = os.getenv(
        "OUTPUT_JSONL", fcfg.get("output_jsonl", "ft_data/qwen4b_crafter_sft_ach.jsonl")
    )
    filters = build_filters()
    # Default: require >1 achievements unless explicitly specified in config/env
    # If caller set REQUIRED_ACHIEVEMENTS or provided config 'required_achievements', we won't override.
    # Otherwise, enforce min achievements via MIN_ACHIEVEMENTS (default 2)
    if not filters.get("required_achievements"):
        try:
            min_ach = int(os.getenv("MIN_ACHIEVEMENTS", str(fcfg.get("min_achievements", 2))))
        except Exception:
            min_ach = 2
        filters["min_achievements"] = min_ach

    window_mode = os.getenv("WINDOW_MODE", "0") == "1"

    print("🤖 Modal/Synth FT Filter (achievements)")
    print("Using database:", db_url)
    print("Output file:", output_path)
    print(
        "Filters:",
        json.dumps(
            {k: (list(v) if isinstance(v, set) else v) for k, v in filters.items()}, indent=2
        ),
    )
    print("Window mode:", window_mode)

    # Print distributions (achievements and rewards) before filtering for visibility
    try:
        import numpy as _np
        from collections import Counter as _Counter

        async with FinetuningDataExtractorV3(db_url) as _ex:
            _sessions = await _ex.get_all_sessions()
            _ach_counts: _Counter[str] = _Counter()
            _rewards: list[float] = []
            for _, _row in _sessions.iterrows():
                _sid = _row["session_id"]
                _ach = await _ex.get_session_achievements(_sid) or []
                for _a in _ach:
                    _ach_counts[_a] += 1
                _met = await _ex.get_session_metrics(_sid)
                try:
                    _rewards.append(float(_met.get("total_reward", 0.0) or 0.0))
                except Exception:
                    pass
            print(f"\nTotal sessions: {len(_sessions)}")
            if _ach_counts:
                print("\nAchievements by session (count):")
                for _name, _c in sorted(_ach_counts.items(), key=lambda x: (-x[1], x[0])):
                    print(f"  {_name}: {_c}")
            if _rewards:
                _r = _np.array(_rewards, dtype=float)
                print("\nReward stats:")
                print(
                    f"  min={_r.min():.2f} median={_np.median(_r):.2f} mean={_r.mean():.2f} max={_r.max():.2f}"
                )
    except Exception:
        pass

    required: set[str] = filters["required_achievements"]
    models: list[str] = filters["models"]
    min_reward: float = filters["min_total_reward"]
    max_cost: float = filters["max_cost"]
    max_tokens: int = filters["max_tokens"]

    stats: dict[str, Any] = {
        "total_sessions": 0,
        "kept_sessions": 0,
        "total_examples": 0,
    }

    async with FinetuningDataExtractorV3(db_url) as extractor:
        all_sessions = await extractor.get_all_sessions()
        stats["total_sessions"] = len(all_sessions)

        kept: list[str] = []
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
                model_query = """
                    SELECT DISTINCT model_name
                    FROM events
                    WHERE session_id = :session_id
                      AND event_type = 'cais'
                      AND model_name IS NOT NULL
                    """
                model_df = await extractor.db_manager.query_traces(
                    model_query, {"session_id": session_id}
                )
                session_models = model_df["model_name"].tolist() if not model_df.empty else []
                if not any(m in models for m in session_models):
                    continue

            # Respect either explicit required achievements OR min_achievements fallback
            min_ach = int(filters.get("min_achievements", 0))
            if required or min_ach > 0:
                achievements = await extractor.get_session_achievements(session_id)
                if not achievements:
                    continue
                if required:
                    if not (required & set(achievements)):
                        continue
                else:
                    if len(achievements) < min_ach:
                        continue

            kept.append(session_id)

        stats["kept_sessions"] = len(kept)

        if window_mode:
            training_data = await extractor.extract_openai_window_format(kept)
        else:
            training_data = await extractor.extract_openai_format(kept)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in training_data:
                f.write(json.dumps(ex) + "\n")
        stats["total_examples"] = len(training_data)

    print(
        "\n✅ Wrote", stats["total_examples"], "examples from", stats["kept_sessions"], "sessions"
    )


if __name__ == "__main__":
    asyncio.run(main())
