#!/usr/bin/env python3
"""Summarise tracing_v3 SQLite data (models, rewards, achievements)."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Row = sqlite3.Row


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_model_usage(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            model_name,
            provider,
            COUNT(*) AS calls,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY model_name, provider
        ORDER BY calls DESC
        """
    ).fetchall()
    stats: list[dict[str, Any]] = []
    for row in rows:
        stats.append(
            {
                "model_name": row["model_name"],
                "provider": row["provider"],
                "calls": int(row["calls"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
            }
        )
    return stats


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


AchievementMap = dict[tuple[str, int], dict[str, list[str]]]


def fetch_achievement_data(
    conn: sqlite3.Connection,
) -> tuple[
    AchievementMap,
    Counter,
    Counter,
    Counter,
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """Return per-turn achievement map and summary counters.

    Returns:
        achievements_map: {(session_id, turn) -> {"unique": [...], "all": [...]}}
        unique_counts_per_session: Counter mapping session -> total unique achievements
        achievement_name_counts: Counter mapping achievement name -> occurrences (unique)
        achievement_size_counts: Counter mapping number of unique achievements per session -> frequency
    """

    achievements_map: AchievementMap = defaultdict(lambda: {"unique": [], "all": []})
    session_unique_sets: dict[str, set[str]] = defaultdict(set)
    session_final_achievements: dict[str, set[str]] = defaultdict(set)
    achievement_name_counts: Counter = Counter()

    # Unique achievements (reward_type = unique_achievement_delta)
    rows = conn.execute(
        """
        SELECT er.session_id, er.reward_value, er.annotation, ev.metadata
        FROM event_rewards er
        JOIN events ev ON er.event_id = ev.id
        WHERE er.reward_type = 'unique_achievement_delta' AND er.reward_value > 0
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        annotation = _parse_json(row["annotation"]) or {}
        metadata = _parse_json(row["metadata"]) or {}
        turn = metadata.get("turn")
        if turn is None:
            continue
        new_unique = annotation.get("new_unique") or []
        if not isinstance(new_unique, list):
            continue
        if new_unique:
            achievements_map[(session_id, int(turn))]["unique"].extend(new_unique)
            session_unique_sets[session_id].update(new_unique)

    # All achievements (reward_type = achievement_delta)
    rows = conn.execute(
        """
        SELECT er.session_id, er.reward_value, er.annotation, ev.metadata
        FROM event_rewards er
        JOIN events ev ON er.event_id = ev.id
        WHERE er.reward_type = 'achievement_delta' AND er.reward_value > 0
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        annotation = _parse_json(row["annotation"]) or {}
        metadata = _parse_json(row["metadata"]) or {}
        turn = metadata.get("turn")
        if turn is None:
            continue
        turned_true = annotation.get("turned_true") or []
        if not isinstance(turned_true, list):
            continue
        if turned_true:
            achievements_map[(session_id, int(turn))]["all"].extend(turned_true)

    # Fallback to outcome rewards metadata to capture final achievements
    rows = conn.execute(
        """
        SELECT session_id, reward_metadata
        FROM outcome_rewards
        WHERE reward_metadata IS NOT NULL
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        metadata = _parse_json(row["reward_metadata"])
        if not isinstance(metadata, dict):
            continue
        final_achievements = metadata.get("achievements") or []
        if isinstance(final_achievements, list):
            cleaned = [a for a in final_achievements if isinstance(a, str)]
            session_unique_sets[session_id].update(cleaned)
            session_final_achievements[session_id].update(cleaned)

    # Build counters from the unique sets
    unique_counts_per_session: Counter = Counter()
    for session_id, achievement_set in session_unique_sets.items():
        unique_counts_per_session[session_id] = len(achievement_set)
        achievement_name_counts.update(achievement_set)

    achievement_size_counts: Counter = Counter()
    for _session_id, count in unique_counts_per_session.items():
        achievement_size_counts[count] += 1

    return (
        achievements_map,
        unique_counts_per_session,
        achievement_name_counts,
        achievement_size_counts,
        session_unique_sets,
        session_final_achievements,
    )


def fetch_reward_summary(conn: sqlite3.Connection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate reward information from outcome_rewards and event_rewards."""

    outcome_row = conn.execute(
        """
        SELECT
            COUNT(*) AS episodes,
            COALESCE(SUM(total_reward), 0) AS total_reward,
            COALESCE(AVG(total_reward), 0) AS avg_reward,
            COALESCE(MIN(total_reward), 0) AS min_reward,
            COALESCE(MAX(total_reward), 0) AS max_reward
        FROM outcome_rewards
        """
    ).fetchone()

    reward_breakdown_rows = conn.execute(
        """
        SELECT
            reward_type,
            COUNT(*) AS events,
            COALESCE(SUM(reward_value), 0) AS total_value,
            COALESCE(AVG(reward_value), 0) AS avg_value
        FROM event_rewards
        GROUP BY reward_type
        ORDER BY events DESC
        """
    ).fetchall()

    breakdown: list[dict[str, Any]] = []
    for row in reward_breakdown_rows:
        breakdown.append(
            {
                "reward_type": row["reward_type"],
                "events": int(row["events"] or 0),
                "total_value": float(row["total_value"] or 0.0),
                "avg_value": float(row["avg_value"] or 0.0),
            }
        )

    outcome = {
        "episodes": int(outcome_row["episodes"] or 0),
        "total_reward": float(outcome_row["total_reward"] or 0.0),
        "avg_reward": float(outcome_row["avg_reward"] or 0.0),
        "min_reward": float(outcome_row["min_reward"] or 0.0),
        "max_reward": float(outcome_row["max_reward"] or 0.0),
    }

    return outcome, breakdown


def format_model_stats(stats: list[dict[str, Any]]) -> str:
    if not stats:
        return "No model usage recorded."
    lines = ["Model usage (by LLM calls):"]
    header = (
        f"{'Model':30} {'Provider':10} {'Calls':>7} {'Tokens (in/out)':>20} {'Avg latency ms':>15}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for item in stats:
        lines.append(
            f"{item['model_name'][:30]:30} "
            f"{(item['provider'] or '')[:10]:10} "
            f"{item['calls']:7d} "
            f"{item['total_tokens']:10d}/{item['output_tokens']:>8d} "
            f"{item['avg_latency_ms']:15.1f}"
        )
    return "\n".join(lines)


def format_achievement_summary(name_counts: Counter, size_counts: Counter) -> str:
    lines = ["Unique achievements unlocked:"]
    if name_counts:
        top = name_counts.most_common()
        for name, count in top:
            lines.append(f"  {name:25} -> {count}")
    else:
        lines.append("  (none recorded)")

    lines.append("")
    lines.append("Sessions bucketed by unique achievement count:")
    if size_counts:
        for size in sorted(size_counts):
            lines.append(f"  {size:2d} unique -> {size_counts[size]} session(s)")
    else:
        lines.append("  (no sessions with achievements)")
    return "\n".join(lines)


def format_reward_summary(outcome: dict[str, Any], breakdown: list[dict[str, Any]]) -> str:
    lines = ["Episode outcome rewards:"]
    episodes = outcome.get("episodes", 0)
    if episodes:
        lines.append(
            f"  Episodes: {episodes} | total={outcome['total_reward']:.2f} | "
            f"avg={outcome['avg_reward']:.2f} | min/max={outcome['min_reward']:.2f}/{outcome['max_reward']:.2f}"
        )
    else:
        lines.append("  (no outcome rewards recorded)")

    lines.append("")
    lines.append("Event reward breakdown (event_rewards table):")
    if breakdown:
        header = f"{'Reward type':20} {'Events':>8} {'Total value':>14} {'Avg value':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in breakdown:
            lines.append(
                f"{row['reward_type'][:20]:20} "
                f"{row['events']:8d} "
                f"{row['total_value']:14.3f} "
                f"{row['avg_value']:12.3f}"
            )
    else:
        lines.append("  (no event rewards recorded)")

    return "\n".join(lines)


def compute_model_achievement_stats(
    conn: sqlite3.Connection, session_unique_sets: dict[str, set[str]]
) -> dict[str, dict[str, Any]]:
    """Aggregate unique-achievement stats per model."""

    rows = conn.execute(
        """
        SELECT session_id, model_name, provider, COUNT(*) AS calls
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY session_id, model_name, provider
        """
    ).fetchall()

    session_models: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        session_id = row["session_id"]
        calls = int(row["calls"] or 0)
        current = session_models.get(session_id)
        if current is None or calls > current[2]:
            session_models[session_id] = (row["model_name"], row["provider"], calls)

    model_stats: dict[str, dict[str, Any]] = {}
    for session_id, (model_name, provider, _calls) in session_models.items():
        achievements = session_unique_sets.get(session_id, set())
        unique_count = len(achievements)

        stats = model_stats.setdefault(
            model_name,
            {
                "providers": set(),
                "sessions": 0,
                "sessions_with_unique": 0,
                "total_unique": 0,
                "max_unique": 0,
                "achievement_counts": Counter(),
            },
        )

        stats["providers"].add(provider or "unknown")
        stats["sessions"] += 1
        stats["total_unique"] += unique_count
        stats["max_unique"] = max(stats["max_unique"], unique_count)
        if unique_count > 0:
            stats["sessions_with_unique"] += 1
            stats["achievement_counts"].update(achievements)

    return model_stats


def format_model_achievement_stats(model_stats: dict[str, dict[str, Any]]) -> str:
    if not model_stats:
        return "Achievement stats by model:\n  (no model sessions recorded)"

    lines = ["Achievement stats by model:"]
    for model_name in sorted(
        model_stats.keys(), key=lambda m: model_stats[m]["sessions"], reverse=True
    ):
        stats = model_stats[model_name]
        providers = ", ".join(sorted(stats["providers"])) if stats["providers"] else "-"
        sessions = stats["sessions"]
        total_unique = stats["total_unique"]
        avg_unique = total_unique / sessions if sessions else 0.0
        sessions_with_unique = stats["sessions_with_unique"]
        max_unique = stats["max_unique"]
        lines.append(
            f"  {model_name} (providers: {providers})\n"
            f"    sessions={sessions} with_unique={sessions_with_unique} "
            f"avg_unique={avg_unique:.2f} max_unique={max_unique}"
        )

        achievement_counts = stats["achievement_counts"]
        if achievement_counts:
            lines.append("    achievements:")
            for name, count in sorted(
                achievement_counts.items(), key=lambda item: item[1], reverse=True
            ):
                lines.append(f"      {name}: {count}")
        else:
            lines.append("    achievements: none")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("traces/v3/synth_ai.db"),
        help="Path to the tracing_v3 SQLite database",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        raise SystemExit(1)

    conn = connect(args.db)
    try:
        model_stats = fetch_model_usage(conn)
        print(format_model_stats(model_stats))
        print("")

        (
            _achievements_map,
            _unique_counts_per_session,
            name_counts,
            size_counts,
            session_unique_sets,
            _session_final_achievements,
        ) = fetch_achievement_data(conn)
        outcome_summary, reward_breakdown = fetch_reward_summary(conn)

        print(format_reward_summary(outcome_summary, reward_breakdown))
        print("")
        print(format_achievement_summary(name_counts, size_counts))
        print("")
        model_achievement_stats = compute_model_achievement_stats(conn, session_unique_sets)
        print(format_model_achievement_stats(model_achievement_stats))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
