#!/usr/bin/env python3
"""Export SFT JSONL from tracing_v3 sqlite using the shared exporter utilities.

Thin wrapper over `examples/warming_up_to_rl/export_trace_sft.py` to keep the
SFT workflow self-contained in this folder while reusing tested logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.warming_up_to_rl.export_trace_sft import (
    build_sft_dataset,
    connect,
    fetch_achievement_data,
    fetch_event_reward_totals,
    fetch_outcome_rewards,
    fetch_session_models,
    parse_event_filters,
    write_jsonl,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("traces/v3/synth_ai.db"))
    p.add_argument("--output", type=Path, default=Path("examples/sft/ft_data/crafter_traces.jsonl"))
    p.add_argument("--model", action="append", dest="models")
    p.add_argument("--provider", action="append", dest="providers")
    p.add_argument("--min-unique", type=int, default=0)
    p.add_argument("--max-unique", type=int, default=None)
    p.add_argument("--exclude-achievement", action="append", dest="exclude_achievements")
    p.add_argument("--require-achievement", action="append", dest="required_achievements")
    p.add_argument("--min-outcome-reward", type=float, default=None)
    p.add_argument("--max-outcome-reward", type=float, default=None)
    p.add_argument("--event-reward", action="append", dest="event_reward_filters")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    conn = connect(args.db)
    try:
        achievements_map, unique_counts, name_counts, size_counts, session_uniques, session_final = (
            fetch_achievement_data(conn)
        )
        session_models = fetch_session_models(conn)
        outcome_data = fetch_outcome_rewards(conn)
        event_totals = fetch_event_reward_totals(conn)
        event_filters = parse_event_filters(args.event_reward_filters)

        allowed_models = set(args.models) if args.models else None
        allowed_providers = set(args.providers) if args.providers else None
        required_achievements = set(args.required_achievements or [])
        excluded_achievements = set(args.exclude_achievements or [])

        eligible: set[str] = set()
        for session_id, (model_name, provider, _calls) in session_models.items():
            if allowed_models and model_name not in allowed_models:
                continue
            if allowed_providers and (provider or "unknown") not in allowed_providers:
                continue

            session_unique = session_uniques.get(session_id, set())
            adjusted_uniques = {a for a in session_unique if a not in excluded_achievements}
            unique_count = len(adjusted_uniques)
            if args.min_unique is not None and unique_count < args.min_unique:
                continue
            if args.max_unique is not None and unique_count > args.max_unique:
                continue

            outcome = outcome_data.get(session_id)
            total_reward = outcome["total_reward"] if outcome else 0.0
            final_achievements = (
                outcome["achievements"] if outcome else session_final.get(session_id, set())
            )
            if args.min_outcome_reward is not None and total_reward < args.min_outcome_reward:
                continue
            if args.max_outcome_reward is not None and total_reward > args.max_outcome_reward:
                continue
            if required_achievements and not required_achievements.issubset(final_achievements):
                continue

            totals = event_totals.get(session_id, {})
            meets_filters = True
            for reward_type, min_total in event_filters:
                total = totals.get(reward_type, {}).get("total", 0.0)
                if total < min_total:
                    meets_filters = False
                    break
            if not meets_filters:
                continue
            eligible.add(session_id)

        if not eligible:
            raise SystemExit("No sessions matched the provided filters.")

        dataset = build_sft_dataset(
            conn,
            achievements_map,
            eligible,
            allowed_models=allowed_models,
            limit=args.limit,
        )
        if not dataset:
            raise SystemExit("No rollout steps matched the filters (after session selection).")

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, dataset)
        print(f"Wrote {len(dataset)} examples -> {args.output}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


