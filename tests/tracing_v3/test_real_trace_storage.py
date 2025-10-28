#!/usr/bin/env python3
"""
Integration-style tests that validate real trace fixtures behave the same
under the libsql-native trace manager.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import MetaData

from synth_ai.tracing_v3.abstractions import (
    EnvironmentEvent,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    SessionTimeStep,
    SessionTrace,
    TimeRecord,
)
from synth_ai.tracing_v3.session_tracer import SessionTracer
from synth_ai.tracing_v3.storage.config import StorageBackend, StorageConfig
from synth_ai.tracing_v3.storage.factory import create_storage
from synth_ai.tracing_v3.turso.native_manager import NativeLibsqlTraceManager
from synth_ai.tracing_v3.constants import TRACE_DB_BASENAME


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces"
SCENARIOS = ("chat_small", "env_rollout", "high_volume")


def _load_manifest(scenario: str) -> tuple[dict, Path]:
    """Load manifest and database path for the given fixture scenario."""
    scenario_dir = FIXTURE_ROOT / scenario
    manifest_path = scenario_dir / "manifest.json"
    db_candidates = sorted(
        scenario_dir.glob(f"{TRACE_DB_BASENAME}*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not db_candidates:
        db_candidates = sorted(scenario_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest for fixture '{scenario}': {manifest_path}")
    if not db_candidates:
        raise RuntimeError(f"Missing database for fixture '{scenario}' in {scenario_dir}")
    db_path = db_candidates[0]
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return manifest, db_path


def _resolve_count(result) -> int:
    """Extract a single integer count from query_traces output."""
    if hasattr(result, "iloc"):
        # pandas DataFrame
        return int(result.iloc[0]["c"])
    if isinstance(result, list):
        return int(result[0]["c"])
    raise TypeError(f"Unsupported result type: {type(result)!r}")


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.asyncio
@pytest.mark.fast
async def test_fixture_counts_align_with_manifest(scenario: str):
    """Ensure all table counts in each fixture match the recorded manifest."""
    manifest, db_path = _load_manifest(scenario)
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        for table, expected in manifest["counts"].items():
            result = await mgr.query_traces(f"SELECT COUNT(*) AS c FROM {table}")
            assert _resolve_count(result) == expected, f"{table} count mismatch"
    finally:
        await mgr.close()


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.asyncio
@pytest.mark.fast
async def test_session_summary_contains_fixture_sessions(scenario: str):
    """Session summary analytics view should surface each fixture session."""
    manifest, db_path = _load_manifest(scenario)
    expected_sessions = set(manifest["session_ids"])
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        summary = await mgr.query_traces(
            "SELECT session_id, num_events, total_cost_usd FROM session_summary"
        )
        if hasattr(summary, "iterrows"):
            sessions = {str(row["session_id"]) for _, row in summary.iterrows()}
        else:
            sessions = {str(row["session_id"]) for row in summary}
        assert sessions == expected_sessions
    finally:
        await mgr.close()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_trace_export_matches_session_count(scenario: str):
    """Sanity check that committed JSONL export lines align with manifest sessions."""
    manifest, _ = _load_manifest(scenario)
    jsonl_path = FIXTURE_ROOT / scenario / "trace_export.jsonl"
    assert jsonl_path.exists(), "Expected JSONL export alongside fixture database"
    line_count = sum(1 for _ in jsonl_path.open("r", encoding="utf-8"))
    assert line_count == len(manifest["session_ids"])


@pytest.mark.asyncio
async def test_env_rollout_contains_environment_events():
    """The env_rollout fixture should retain environment event rows."""
    manifest, db_path = _load_manifest("env_rollout")
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        result = await mgr.query_traces(
            "SELECT COUNT(*) AS c FROM events WHERE event_type = 'environment'"
        )
        assert _resolve_count(result) > 0, "Expected environment events in env_rollout fixture"
        # Reward tallies should line up with manifest counts
        rewards = await mgr.query_traces("SELECT COUNT(*) AS c FROM event_rewards")
        assert _resolve_count(rewards) == manifest["counts"]["event_rewards"]
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_high_volume_fixture_reflects_peak_event_load():
    """High volume fixture should represent the session with the most events."""
    manifest, db_path = _load_manifest("high_volume")
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        events = await mgr.query_traces("SELECT COUNT(*) AS c FROM events")
        assert _resolve_count(events) == manifest["counts"]["events"]
        # Ensure there are multiple CAIS events to stress LLM pathways
        cais = await mgr.query_traces("SELECT COUNT(*) AS c FROM events WHERE event_type = 'cais'")
        assert _resolve_count(cais) > 0, "Expected LLM CAIS events in high_volume fixture"
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_get_session_trace_returns_timesteps_consistent_with_counts():
    """Ensure get_session_trace returns structured data matching stored counts."""
    manifest, db_path = _load_manifest("high_volume")
    session_id = manifest["session_ids"][0]
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        data = await mgr.get_session_trace(session_id)
        assert data is not None
        assert data["session_id"] == session_id
        assert data["num_timesteps"] == len(data["timesteps"])
        assert data["num_events"] == manifest["counts"]["events"]
        assert data["num_messages"] == manifest["counts"]["messages"]
        # Ensure timesteps are ordered consistently
        indices = [step["step_index"] for step in data["timesteps"]]
        assert indices == sorted(indices)
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_get_model_usage_reports_cais_activity():
    """Model usage analytics should surface entries for fixtures with CAIS events."""
    manifest, db_path = _load_manifest("high_volume")
    mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{db_path}")
    await mgr.initialize()
    try:
        model_usage = await mgr.get_model_usage()
        if hasattr(model_usage, "iterrows"):
            total = int(model_usage["usage_count"].sum())
        else:
            total = sum(int(row["usage_count"] or 0) for row in model_usage)
        assert total > 0, "Expected model usage stats to contain at least one row"
        # Ensure session IDs referenced appear in manifest
        events = await mgr.query_traces(
            "SELECT DISTINCT session_id FROM events WHERE event_type='cais'"
        )
        if hasattr(events, "iterrows"):
            session_ids = {str(row["session_id"]) for _, row in events.iterrows()}
        else:
            session_ids = {str(row["session_id"]) for row in events}
        assert set(manifest["session_ids"]).issuperset(session_ids)
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_storage_factory_handles_fixture_connection_string():
    """Storage factory should return a native manager for fixture URLs."""
    _, db_path = _load_manifest("chat_small")
    config = StorageConfig(
        backend=StorageBackend.SQLITE,
        connection_string=f"sqlite+aiosqlite:///{db_path}",
    )
    storage = create_storage(config)
    assert isinstance(storage, NativeLibsqlTraceManager)
    await storage.initialize()
    try:
        result = await storage.query_traces("SELECT COUNT(*) AS c FROM session_traces")
        assert _resolve_count(result) == 3
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_storage_factory_defaults_to_native():
    _, db_path = _load_manifest("chat_small")
    config = StorageConfig(connection_string=f"sqlite+aiosqlite:///{db_path}")
    storage = create_storage(config)
    assert isinstance(storage, NativeLibsqlTraceManager)
    await storage.initialize()
    try:
        result = await storage.query_traces("SELECT COUNT(*) AS c FROM session_traces")
        assert _resolve_count(result) == 3
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.fast
async def test_session_tracer_uses_native_by_default(tmp_path):
    """SessionTracer should instantiate the native manager by default."""
    temp_db = tmp_path / "native_session.db"
    tracer = SessionTracer(db_url=f"sqlite+aiosqlite:///{temp_db}")
    await tracer.initialize()
    try:
        assert isinstance(tracer.db, NativeLibsqlTraceManager)
    finally:
        await tracer.close()


@pytest.mark.asyncio
@pytest.mark.fast
async def test_native_model_usage_matches_sqlite_view():
    """Native query helpers should produce the same results as the underlying view."""
    _, db_path = _load_manifest("high_volume")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        legacy_records = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM model_usage_stats ORDER BY usage_count DESC"
            )
        ]
    finally:
        conn.close()

    native_storage = create_storage(StorageConfig(connection_string=f"sqlite+aiosqlite:///{db_path}"))
    assert isinstance(native_storage, NativeLibsqlTraceManager)
    await native_storage.initialize()
    try:
        native_usage = await native_storage.get_model_usage()
    finally:
        await native_storage.close()

    native_records = sorted(_as_records(native_usage), key=lambda r: (r["model_name"], r["provider"]))
    assert legacy_records == native_records


@pytest.mark.asyncio
async def test_native_get_session_trace_matches_sqlite():
    manifest, db_path = _load_manifest("high_volume")
    session_id = manifest["session_ids"][0]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session_row = conn.execute(
            """
            SELECT session_id, created_at, num_timesteps, num_events, num_messages, metadata
            FROM session_traces
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        assert session_row is not None

        step_rows = conn.execute(
            """
            SELECT step_id, step_index, turn_number, started_at, completed_at, step_metadata
            FROM session_timesteps
            WHERE session_id = ?
            ORDER BY step_index ASC
            """,
            (session_id,),
        ).fetchall()
        legacy_data = {
            "session_id": session_row["session_id"],
            "created_at": datetime.fromisoformat(session_row["created_at"]),
            "num_timesteps": session_row["num_timesteps"],
            "num_events": session_row["num_events"],
            "num_messages": session_row["num_messages"],
            "metadata": json.loads(session_row["metadata"])
            if session_row["metadata"]
            else {},
            "timesteps": [
                {
                    "step_id": row["step_id"],
                    "step_index": row["step_index"],
                    "turn_number": row["turn_number"],
                    "started_at": datetime.fromisoformat(row["started_at"])
                    if row["started_at"]
                    else None,
                    "completed_at": datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"]
                    else None,
                    "metadata": json.loads(row["step_metadata"])
                    if row["step_metadata"]
                    else {},
                }
                for row in step_rows
            ],
        }
    finally:
        conn.close()

    native_storage = create_storage(StorageConfig(connection_string=f"sqlite+aiosqlite:///{db_path}"))
    assert isinstance(native_storage, NativeLibsqlTraceManager)
    await native_storage.initialize()
    try:
        native_data = await native_storage.get_session_trace(session_id)
    finally:
        await native_storage.close()

    assert legacy_data is not None and native_data is not None
    assert _normalise_datetimes(legacy_data) == _normalise_datetimes(native_data)


@pytest.mark.asyncio
@pytest.mark.fast
async def test_native_get_sessions_by_experiment_matches_legacy(tmp_path):
    db_path = tmp_path / "experiments.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracer = SessionTracer(storage=NativeLibsqlTraceManager(db_url=db_url))
    await tracer.initialize()
    try:
        await tracer.db.create_experiment("exp-123", name="exp", description=None, configuration=None)
        for idx in range(2):
            session_id = f"exp_session_{idx}"
            await tracer.start_session(session_id=session_id, metadata={"idx": idx})
            await tracer.end_session()
            await tracer.db.link_session_to_experiment(session_id, "exp-123")
    finally:
        await tracer.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        legacy_sessions = [
            {
                "session_id": row["session_id"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "num_timesteps": row["num_timesteps"],
                "num_events": row["num_events"],
                "num_messages": row["num_messages"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in conn.execute(
                """
                SELECT session_id, created_at, num_timesteps, num_events, num_messages, metadata
                FROM session_traces
                WHERE experiment_id = ?
                ORDER BY created_at DESC
                """,
                ("exp-123",),
            )
        ]
    finally:
        conn.close()

    native_storage = create_storage(StorageConfig(connection_string=db_url))
    assert isinstance(native_storage, NativeLibsqlTraceManager)
    await native_storage.initialize()
    try:
        native_sessions = await native_storage.get_sessions_by_experiment("exp-123")
    finally:
        await native_storage.close()

    assert _normalise_datetimes({"sessions": legacy_sessions}) == _normalise_datetimes(
        {"sessions": native_sessions}
    )


@pytest.mark.asyncio
@pytest.mark.fast
async def test_native_delete_session_removes_rows(tmp_path):
    db_path = tmp_path / "delete_me.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracer = SessionTracer(storage=NativeLibsqlTraceManager(db_url=db_url))
    await tracer.initialize()
    try:
        await tracer.start_session(session_id="delete_me", metadata={"target": True})
        await tracer.start_timestep("step-delete")
        await tracer.record_message("test", "assistant")
        await tracer.record_event(
            EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=1.234))
        )
        await tracer.end_session()
    finally:
        await tracer.close()

    native_storage = create_storage(StorageConfig(connection_string=db_url))
    assert isinstance(native_storage, NativeLibsqlTraceManager)
    await native_storage.initialize()
    try:
        assert await native_storage.delete_session("delete_me") is True
        assert await native_storage.delete_session("delete_me") is False
    finally:
        await native_storage.close()

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM session_traces WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_timesteps WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM event_rewards WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM outcome_rewards WHERE session_id='delete_me'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.fast
async def test_native_insert_session_trace_idempotent(tmp_path):
    db_path = tmp_path / "native_insert.db"
    storage = create_storage(StorageConfig(connection_string=f"sqlite+aiosqlite:///{db_path}"))
    assert isinstance(storage, NativeLibsqlTraceManager)
    await storage.initialize()

    trace = SessionTrace(
        session_id="native-session",
        created_at=datetime.now(UTC),
        session_time_steps=[
            SessionTimeStep(step_id="step-1", step_index=0, turn_number=1, step_metadata={"a": 1})
        ],
        event_history=[
            EnvironmentEvent(
                system_instance_id="env",
                time_record=TimeRecord(event_time=1.23),
                reward=0.5,
                metadata={"step_id": "step-1"},
            )
        ],
        markov_blanket_message_history=[
            SessionEventMarkovBlanketMessage(
                content=SessionMessageContent(text="hi"),
                message_type="assistant",
                time_record=TimeRecord(event_time=1.25),
                metadata={"step_id": "step-1"},
            )
        ],
        metadata={"source": "native"},
    )

    await storage.insert_session_trace(trace)
    # Second call should be idempotent (no duplicate rows)
    await storage.insert_session_trace(trace)
    await storage.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        assert conn.execute(
            "SELECT COUNT(*) FROM session_traces WHERE session_id='native-session'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM session_timesteps WHERE session_id='native-session'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id='native-session'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='native-session'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.fast
async def test_session_tracer_native_write_pipeline(tmp_path):
    """Native manager should persist incremental writes end-to-end."""
    db_path = tmp_path / "native_pipeline.db"
    tracer = SessionTracer(db_url=f"sqlite+aiosqlite:///{db_path}")
    await tracer.initialize()

    session_id = await tracer.start_session(metadata={"source": "native"})
    await tracer.start_timestep("step-1", turn_number=0)

    event = EnvironmentEvent(
        system_instance_id="env",
        time_record=TimeRecord(event_time=1234.5),
        reward=1.5,
        metadata={"note": "native-path"},
        system_state_after={"obs": "ok"},
    )
    event_id = await tracer.record_event(event)
    message_id = await tracer.record_message("hello", "assistant")

    assert event_id is not None
    assert message_id is not None

    reward_id = await tracer.record_event_reward(
        event_id=event_id,
        reward_value=0.25,
        reward_type="shaped",
        key="test-reward",
        annotation={"details": "native"},
    )
    outcome_id = await tracer.record_outcome_reward(
        total_reward=3,
        achievements_count=1,
        total_steps=1,
        reward_metadata={"summary": "done"},
    )
    assert reward_id is not None
    assert outcome_id is not None

    await tracer.end_session()
    await tracer.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session_row = conn.execute(
            "SELECT num_events, num_messages FROM session_traces WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert session_row["num_events"] == 1
        assert session_row["num_messages"] == 1

        event_row = conn.execute("SELECT event_type, reward FROM events").fetchone()
        assert event_row["event_type"] == "environment"
        assert pytest.approx(event_row["reward"], rel=1e-3) == 1.5

        message_row = conn.execute("SELECT message_type, content FROM messages").fetchone()
        assert message_row["message_type"] == "assistant"
        assert message_row["content"] == "hello"

        reward_rows = conn.execute("SELECT reward_type, key FROM event_rewards").fetchall()
        assert any(row["key"] == "test-reward" and row["reward_type"] == "shaped" for row in reward_rows)

        outcome_row = conn.execute("SELECT total_reward FROM outcome_rewards").fetchone()
        assert outcome_row["total_reward"] == 3
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.fast
async def test_session_tracer_replays_env_fixture_subset(tmp_path):
    """Replay a subset of an environment session via SessionTracer and verify persistence."""
    manifest, fixture_db = _load_manifest("env_rollout")
    conn = sqlite3.connect(fixture_db)
    conn.row_factory = sqlite3.Row
    try:
        source_session = manifest["session_ids"][0]
        timesteps = conn.execute(
            """
            SELECT id, step_id, step_index, turn_number, step_metadata
            FROM session_timesteps
            WHERE session_id = ?
            ORDER BY step_index
            LIMIT 3
            """,
            (source_session,),
        ).fetchall()

        temp_db = tmp_path / "replay.db"
        tracer = SessionTracer(db_url=f"sqlite+aiosqlite:///{temp_db}")
        await tracer.initialize()
        await tracer.start_session(metadata={"source": "fixture_replay"})

        total_events = 0
        total_messages = 0

        for step in timesteps:
            metadata = json.loads(step["step_metadata"]) if step["step_metadata"] else {}
            await tracer.start_timestep(
                step["step_id"], turn_number=step["turn_number"], metadata=metadata
            )

            events = conn.execute(
                """
                SELECT *
                FROM events
                WHERE session_id = ? AND timestep_id = ? AND event_type = 'environment'
                ORDER BY id
                LIMIT 2
                """,
                (source_session, step["id"]),
            ).fetchall()
            for row in events:
                event_metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                state_before = (
                    json.loads(row["system_state_before"])
                    if row["system_state_before"]
                    else None
                )
                state_after = (
                    json.loads(row["system_state_after"]) if row["system_state_after"] else None
                )
                event = EnvironmentEvent(
                    system_instance_id=row["system_instance_id"] or "",
                    time_record=TimeRecord(
                        event_time=float(row["event_time"] or 0.0),
                        message_time=row["message_time"],
                    ),
                    metadata=event_metadata,
                    reward=float(row["reward"] or 0.0),
                    terminated=bool(row["terminated"]),
                    truncated=bool(row["truncated"]),
                    system_state_before=state_before,
                    system_state_after=state_after,
                )
                await tracer.record_event(event)
                total_events += 1

            messages = conn.execute(
                """
                SELECT *
                FROM messages
                WHERE session_id = ? AND timestep_id = ?
                ORDER BY id
                LIMIT 2
                """,
                (source_session, step["id"]),
            ).fetchall()
            for msg in messages:
                try:
                    content = json.loads(msg["content"])
                except Exception:
                    content = msg["content"]
                metadata = json.loads(msg["metadata"]) if msg["metadata"] else {}
                await tracer.record_message(
                    content,
                    msg["message_type"],
                    event_time=float(msg["event_time"] or 0.0),
                    message_time=msg["message_time"],
                    metadata=metadata,
                )
                total_messages += 1

            await tracer.end_timestep(step["step_id"])

        replay_trace = await tracer.end_session()
        assert len(replay_trace.event_history) == total_events
        assert len(replay_trace.session_time_steps) == len(timesteps)
        assert len(replay_trace.markov_blanket_message_history) == total_messages

        if tracer.db:
            await tracer.db.close()

        mgr = NativeLibsqlTraceManager(f"sqlite+aiosqlite:///{temp_db}")
        await mgr.initialize()
        try:
            assert _resolve_count(await mgr.query_traces("SELECT COUNT(*) AS c FROM session_traces")) == 1
            assert _resolve_count(await mgr.query_traces("SELECT COUNT(*) AS c FROM session_timesteps")) == len(
                timesteps
            )
            assert _resolve_count(await mgr.query_traces("SELECT COUNT(*) AS c FROM events")) == total_events
            assert _resolve_count(await mgr.query_traces("SELECT COUNT(*) AS c FROM messages")) == total_messages
        finally:
            await mgr.close()
    finally:
        conn.close()
def _as_records(result) -> list[dict]:
    """Normalise query results to a deterministic list of dicts."""
    if hasattr(result, "to_dict"):
        return result.to_dict("records")  # type: ignore[call-arg]
    if isinstance(result, list):
        return [dict(row) for row in result]
    raise TypeError(f"Unsupported result container: {type(result)!r}")


def _normalise_datetimes(payload):
    def _convert(value):
        if isinstance(value, MetaData):
            return {}
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return _convert(payload)
