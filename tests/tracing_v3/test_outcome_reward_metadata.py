import asyncio
import json
import os
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_session_tracer_persists_reward_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Arrange: isolate a fresh sqlite DB path
    db_path = tmp_path / "synth_ai.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("TURSO_LOCAL_DB_URL", db_url)

    # Lazy import after env set so tracing picks up the DB URL
    from synth_ai.tracing_v3 import SessionTracer  # type: ignore

    tracer = SessionTracer(auto_save=True, db_url=db_url)

    # Act: write a minimal session with an outcome containing reward_metadata
    async with tracer.session(metadata={"test": True}):
        outcome_id = await tracer.record_outcome_reward(
            total_reward=2,
            achievements_count=2,
            total_steps=7,
            reward_metadata={
                "achievements": ["collect_wood", "wake_up"],
                "catalog": ["collect_wood", "wake_up", "place_table"],
            },
        )
        assert isinstance(outcome_id, int)

    # Assert: query raw DB to verify JSON field round-trips
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT total_reward, achievements_count, total_steps, reward_metadata FROM outcome_rewards ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        total_reward, achievements_count, total_steps, reward_metadata = row
        assert total_reward == 2
        assert achievements_count == 2
        assert total_steps == 7
        # reward_metadata stored as JSON text
        md = json.loads(reward_metadata) if isinstance(reward_metadata, str) else reward_metadata
        assert md["achievements"] == ["collect_wood", "wake_up"]
        assert "catalog" in md
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_manager_insert_outcome_reward_accepts_reward_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Arrange
    db_path = tmp_path / "synth_ai.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("TURSO_LOCAL_DB_URL", db_url)

    from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager  # type: ignore
    from synth_ai.tracing_v3.abstractions import SessionTrace  # type: ignore
    from sqlalchemy import text
    from datetime import datetime

    mgr = AsyncSQLTraceManager(db_url)
    await mgr.initialize()

    # Create a minimal session row for FK
    async with mgr.session() as sess:
        await sess.execute(
            text(
                "INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata) VALUES (:sid, :created_at, 0, 0, 0, '{}')"
            ),
            {"sid": "sess_test", "created_at": datetime.utcnow()},
        )
        await sess.commit()

    # Act: insert outcome with reward_metadata
    oid = await mgr.insert_outcome_reward(
        "sess_test",
        total_reward=1,
        achievements_count=1,
        total_steps=5,
        reward_metadata={"achievements": ["collect_wood"]},
    )
    assert isinstance(oid, int)

    # Assert
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT reward_metadata FROM outcome_rewards WHERE id=?",
            (oid,),
        ).fetchone()
        assert row and row[0]
        md = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        assert md["achievements"] == ["collect_wood"]
    finally:
        conn.close()


