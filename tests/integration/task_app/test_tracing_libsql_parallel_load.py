from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime

import pytest

from synth_ai.tracing_v3.session_tracer import SessionTracer


@pytest.mark.asyncio
async def test_libsql_tracing_handles_parallel_rollouts(monkeypatch, tmp_path):
    """Verify that the libSQL backend sustains concurrent session writes."""

    db_path = tmp_path / "libsql_parallel.sqlite"

    # Ensure fresh environment so libSQL path is selected
    for key in (
        "SYNTH_TRACES_DB",
        "SYNTH_TRACES_DIR",
        "LIBSQL_URL",
        "LIBSQL_AUTH_TOKEN",
        "TURSO_DATABASE_URL",
        "TURSO_AUTH_TOKEN",
        "TURSO_LOCAL_DB_URL",
        "SQLD_DB_PATH",
        "TURSO_NATIVE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("LIBSQL_URL", "libsql://parallel-test")

    def _connect_stub(database: str, **kwargs):
        conn = sqlite3.connect(
            db_path,
            timeout=30,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    monkeypatch.setattr(
        "synth_ai.tracing_v3.turso.native_manager._libsql_connect",
        _connect_stub,
    )

    async def _run_session(idx: int) -> str:
        tracer = SessionTracer(auto_save=True)
        session_id = await tracer.start_session(metadata={"worker": idx})
        await tracer.start_timestep(f"step-{idx}", turn_number=idx)
        await tracer.record_message(
            content=f"worker-{idx}",
            message_type="system",
            event_time=datetime.now().timestamp(),
        )
        await tracer.end_timestep()
        await tracer.end_session(save=True)
        if tracer.db:
            await tracer.db.close()
        return session_id

    session_ids = await asyncio.gather(*(_run_session(i) for i in range(8)))
    assert len(session_ids) == 8
    assert len(set(session_ids)) == 8

    with sqlite3.connect(db_path) as verify_conn:
        verify_conn.row_factory = sqlite3.Row
        count = verify_conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
        assert count == 8

