#!/usr/bin/env python3
"""
Trace analysis utilities for Crafter v3 traces (sqld/Turso).

Subcommands:
  - list:    List achievements present in the database and counts
  - filter:  Filter sessions by required achievements and export OpenAI-format JSONL
  - stats:   Compare rewards and achievement frequencies for filtered vs. others

Usage examples:
  uvpm examples.evals.trace_analysis list --db traces/v3/synth_ai.db/dbs/default/data
  uvpm examples.evals.trace_analysis filter --db traces/v3/synth_ai.db/dbs/default/data \
      --achievements collect_wood --output ft_data/evals_collect_wood.jsonl
"""

import argparse
import asyncio
import json
import json as pyjson
import math
from pathlib import Path

from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso import (
    FinetuningDataExtractorV3,
)


def build_db_url(path: str) -> str:
    if path.startswith("sqlite+"):
        return path
    return f"sqlite+aiosqlite:///{path}"


async def cmd_list(db_path: str) -> None:
    db_url = build_db_url(db_path)
    async with FinetuningDataExtractorV3(db_url) as ex:
        sessions = await ex.get_all_sessions()
        achievement_counts: dict[str, int] = {}
        for _, row in sessions.iterrows():
            ach_list = await ex.get_session_achievements(row["session_id"]) or []
            for name in ach_list:
                achievement_counts[name] = achievement_counts.get(name, 0) + 1

        print("Achievements present (session counts):")
        for name in sorted(achievement_counts.keys()):
            print(f"  - {name}: {achievement_counts[name]}")


async def cmd_filter(
    db_path: str, achievements: list[str], output: str, models: list[str] | None = None
) -> None:
    db_url = build_db_url(db_path)
    required: set[str] = set(achievements)
    async with FinetuningDataExtractorV3(db_url) as ex:
        sessions = await ex.get_all_sessions()
        kept: list[str] = []
        for _, row in sessions.iterrows():
            if models:
                # Restrict to sessions containing any of the requested models
                model_df = await ex.db_manager.query_traces(
                    """
                    SELECT DISTINCT model_name
                    FROM events
                    WHERE session_id = :session_id
                      AND event_type = 'cais'
                      AND model_name IS NOT NULL
                    """,
                    {"session_id": row["session_id"]},
                )
                session_models = (
                    model_df["model_name"].tolist()
                    if model_df is not None and not model_df.empty
                    else []
                )
                if not any(m in session_models for m in models):
                    continue
            ach_list = await ex.get_session_achievements(row["session_id"]) or []
            if required & set(ach_list):
                kept.append(row["session_id"])

        data = await ex.extract_openai_format(kept)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for exm in data:
                f.write(json.dumps(exm) + "\n")
        print(f"✅ Wrote {len(data)} examples from {len(kept)} sessions → {output}")


async def _first_achievement_step(
    ex: FinetuningDataExtractorV3, session_id: str, required: set[str]
) -> int | None:
    q = """
        SELECT message_time, system_state_after
        FROM events
        WHERE session_id = :session_id
          AND event_type = 'environment'
        ORDER BY message_time ASC
        """
    df = await ex.db_manager.query_traces(q, {"session_id": session_id})
    if df is None or df.empty:
        return None
    seen: set[str] = set()
    for _, row in df.iterrows():
        st = row.get("system_state_after")
        if isinstance(st, str):
            try:
                st = pyjson.loads(st)
            except Exception:
                st = None
        ach = None
        if isinstance(st, dict):
            ps = st.get("public_state") or {}
            ach = ps.get("achievements_status") or {}
        if isinstance(ach, dict):
            for name, unlocked in ach.items():
                if unlocked and name in required and name not in seen:
                    return int(row.get("message_time") or 0)
    return None


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    var = sum((v - m) * (v - m) for v in values) / len(values)
    return math.sqrt(var)


async def cmd_stats(db_path: str, achievements: list[str], models: list[str] | None = None) -> None:
    db_url = build_db_url(db_path)
    required: set[str] = set(achievements)
    async with FinetuningDataExtractorV3(db_url) as ex:
        sessions = await ex.get_all_sessions()
        matched_rewards: list[float] = []
        other_rewards: list[float] = []
        first_steps: list[int] = []
        matched_count: int = 0
        other_count: int = 0
        matched_ach_counts: dict[str, int] = {}
        other_ach_counts: dict[str, int] = {}

        for _, row in sessions.iterrows():
            sid = row["session_id"]
            if models:
                model_df = await ex.db_manager.query_traces(
                    """
                    SELECT DISTINCT model_name
                    FROM events
                    WHERE session_id = :session_id
                      AND event_type = 'cais'
                      AND model_name IS NOT NULL
                    """,
                    {"session_id": sid},
                )
                session_models = (
                    model_df["model_name"].tolist()
                    if model_df is not None and not model_df.empty
                    else []
                )
                if not any(m in session_models for m in models):
                    continue

            ach_list = await ex.get_session_achievements(sid) or []
            metrics = await ex.get_session_metrics(sid)
            reward = float(metrics.get("total_reward", 0.0))

            if required & set(ach_list):
                matched_rewards.append(reward)
                step = await _first_achievement_step(ex, sid, required)
                if step is not None:
                    first_steps.append(step)
                matched_count += 1
                for name in ach_list:
                    matched_ach_counts[name] = matched_ach_counts.get(name, 0) + 1
            else:
                other_rewards.append(reward)
                other_count += 1
                for name in ach_list:
                    other_ach_counts[name] = other_ach_counts.get(name, 0) + 1

        print("Matched sessions (any of:", ", ".join(sorted(required)), ")")
        print(
            f"  n={len(matched_rewards)}  avg_reward={_mean(matched_rewards):.2f}  stddev={_stddev(matched_rewards):.2f}"
        )
        if first_steps:
            print(
                f"  avg_first_unlock_step={_mean([float(s) for s in first_steps]):.1f}  stddev={_stddev([float(s) for s in first_steps]):.1f}"
            )
        else:
            print("  avg_first_unlock_step=n/a (no unlocks recorded)")
        print("Others")
        print(
            f"  n={len(other_rewards)}  avg_reward={_mean(other_rewards):.2f}  stddev={_stddev(other_rewards):.2f}"
        )

        # Achievement frequency comparison (by session presence), excluding required achievements
        all_achievements: set[str] = set(matched_ach_counts.keys()) | set(other_ach_counts.keys())
        compare_achievements = [a for a in sorted(all_achievements) if a not in required]
        if compare_achievements and (matched_count > 0 or other_count > 0):
            print("\nAchievement frequency by session (matched vs others):")
            # Build rows with absolute percentage difference for sorting
            rows: list[tuple[float, str, int, float, int, float]] = []
            for name in compare_achievements:
                m_n = matched_ach_counts.get(name, 0)
                o_n = other_ach_counts.get(name, 0)
                m_pct = (m_n / matched_count * 100.0) if matched_count else 0.0
                o_pct = (o_n / other_count * 100.0) if other_count else 0.0
                diff = abs(m_pct - o_pct)
                rows.append((diff, name, m_n, m_pct, o_n, o_pct))

            # Show top 10 differences
            rows.sort(reverse=True)
            limit = min(10, len(rows))
            for i in range(limit):
                _, name, m_n, m_pct, o_n, o_pct = rows[i]
                print(
                    f"  - {name}: matched {m_n}/{matched_count} ({m_pct:.1f}%), others {o_n}/{other_count} ({o_pct:.1f}%)"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter v3 trace analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List achievements present in DB")
    p_list.add_argument(
        "--db", required=True, help="Path to sqld internal data file or full sqlite+aiosqlite URL"
    )

    p_filter = sub.add_parser("filter", help="Filter sessions by achievements and export JSONL")
    p_filter.add_argument(
        "--db", required=True, help="Path to sqld internal data file or full sqlite+aiosqlite URL"
    )
    p_filter.add_argument(
        "--achievements",
        nargs="+",
        required=True,
        help="Required achievements (any match keeps session)",
    )
    p_filter.add_argument("--output", required=True, help="Output JSONL path")
    p_filter.add_argument("--models", nargs="*", help="Optional model names to include (any match)")

    p_stats = sub.add_parser("stats", help="Show summary stats for filtered vs others")
    p_stats.add_argument(
        "--db", required=True, help="Path to sqld internal data file or full sqlite+aiosqlite URL"
    )
    p_stats.add_argument(
        "--achievements", nargs="+", required=True, help="Achievements to match (any match)"
    )
    p_stats.add_argument("--models", nargs="*", help="Optional model names to include (any match)")

    args = parser.parse_args()

    if args.command == "list":
        asyncio.run(cmd_list(args.db))
        return
    if args.command == "filter":
        asyncio.run(cmd_filter(args.db, args.achievements, args.output, args.models or None))
        return
    if args.command == "stats":
        asyncio.run(cmd_stats(args.db, args.achievements, args.models or None))
        return


if __name__ == "__main__":
    main()
