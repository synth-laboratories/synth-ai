#!/usr/bin/env python3
"""
Verify reward persistence in a traces database.

Usage:
  uv run python -m synth_ai.scripts.verify_rewards --db /path/to/db.sqlite --min-reward 1
"""

import argparse
import asyncio
import os
from typing import Dict

from sqlalchemy import text

from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager


async def verify(db_path: str, min_reward: int) -> int:
    db_url = db_path
    if not db_url.startswith("sqlite+aiosqlite:///"):
        db_url = f"sqlite+aiosqlite:///{os.path.abspath(db_path)}"

    mgr = AsyncSQLTraceManager(db_url=db_url)
    await mgr.initialize()

    try:
        async with mgr.session() as session:
            # Sessions with outcome_rewards
            q_good = text(
                """
                SELECT session_id, MAX(total_reward) as total_reward
                FROM outcome_rewards
                GROUP BY session_id
                """
            )
            res = await session.execute(q_good)
            outcomes = {row[0]: int(row[1]) for row in res.fetchall()}

            # Sessions without outcome_rewards
            q_missing = text(
                """
                SELECT s.session_id
                FROM session_traces s
                LEFT JOIN outcome_rewards o ON s.session_id = o.session_id
                WHERE o.session_id IS NULL
                """
            )
            res2 = await session.execute(q_missing)
            missing = [row[0] for row in res2.fetchall()]

            # Aggregate event_rewards per session (informational)
            q_event = text(
                """
                SELECT session_id, COALESCE(SUM(reward_value), 0.0) as sum_rewards
                FROM event_rewards
                GROUP BY session_id
                """
            )
            res3 = await session.execute(q_event)
            event_sums: Dict[str, float] = {row[0]: float(row[1]) for row in res3.fetchall()}

        print(f"Sessions with outcome_rewards: {len(outcomes)}")
        print(f"Sessions missing outcome_rewards: {len(missing)}")
        if missing:
            print("Missing session_ids:", ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))

        # Threshold check
        qualifying = {sid: r for sid, r in outcomes.items() if r >= min_reward}
        print(f"Sessions with total_reward >= {min_reward}: {len(qualifying)}")

        # Show a small comparison snapshot
        sample = list(qualifying.items())[:5]
        for sid, tot in sample:
            er = event_sums.get(sid, 0.0)
            print(f"  {sid}: outcome={tot}, sum(event_rewards)={er:.2f}")

        # Exit non-zero if any sessions are missing outcome rewards
        if missing:
            return 2
        if min_reward > 0 and not qualifying:
            return 3
        return 0
    finally:
        await mgr.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify reward persistence in traces DB")
    ap.add_argument("--db", required=True, help="Path to traces SQLite DB (aiosqlite)")
    ap.add_argument("--min-reward", type=int, default=0, help="Minimum total_reward to consider qualifying")
    args = ap.parse_args()

    return asyncio.run(verify(args.db, args.min_reward))


if __name__ == "__main__":
    raise SystemExit(main())


