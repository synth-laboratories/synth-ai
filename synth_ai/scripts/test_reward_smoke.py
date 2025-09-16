#!/usr/bin/env python3
"""
Create a minimal traces DB with one session, one environment event (with reward),
and an outcome reward row; used to smoke-test reward persistence and conversion.

Usage:
  uv run python -m synth_ai.scripts.test_reward_smoke --db /abs/path/to/test_rewards.db
"""

import argparse
import asyncio
import os
from pathlib import Path
from datetime import datetime

from synth_ai.tracing_v3 import SessionTracer
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager
from synth_ai.tracing_v3.abstractions import EnvironmentEvent, TimeRecord


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to a sqlite db file to create")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    mgr = AsyncSQLTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()

    tracer = SessionTracer()
    tracer.db = mgr

    session_id = f"smoke_{int(datetime.utcnow().timestamp())}"
    async with tracer.session(session_id=session_id, metadata={"purpose": "reward_smoke"}):
        await tracer.start_timestep("step_1", turn_number=0)
        await tracer.record_message(content="{\"observation\": {\"inventory\": {}, \"achievements_status\": {}}}", message_type="system")

        event = EnvironmentEvent(
            system_instance_id="smoke_env",
            time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
            reward=1.0,
            terminated=False,
            truncated=False,
            system_state_before={"dummy": True},
            system_state_after={"dummy": True},
            metadata={"note": "smoke event"},
        )
        await tracer.record_event(event)

        await tracer.record_message(content="{\"result\": true, \"reward\": 1}", message_type="system")
        await tracer.record_outcome_reward(total_reward=1, achievements_count=1, total_steps=1)

    await mgr.close()
    print(f"✅ Wrote smoke traces DB: {db_path}")
    print(f"Session id: {session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


