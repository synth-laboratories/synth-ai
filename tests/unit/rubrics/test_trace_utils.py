from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from synth_ai.core.tracing_v3.trace_utils import (
    DeterministicMetrics,
    compute_deterministic_metrics,
    fetch_crafter_sessions,
    load_session_trace,
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE session_traces (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            metadata TEXT
        );

        CREATE TABLE session_timesteps (
            session_id TEXT,
            step_id TEXT,
            step_index INTEGER,
            turn_number INTEGER,
            started_at TEXT,
            completed_at TEXT,
            step_metadata TEXT
        );

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT,
            system_instance_id TEXT,
            event_time TEXT,
            message_time TEXT,
            created_at TEXT,
            model_name TEXT,
            provider TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd REAL,
            latency_ms REAL,
            span_id TEXT,
            trace_id TEXT,
            call_records TEXT,
            reward REAL,
            terminated INTEGER,
            truncated INTEGER,
            system_state_before TEXT,
            system_state_after TEXT,
            metadata TEXT,
            event_metadata TEXT
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_type TEXT,
            content TEXT,
            event_time TEXT,
            message_time TEXT,
            timestamp TEXT,
            metadata TEXT
        );

        CREATE TABLE event_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_id INTEGER,
            turn_number INTEGER,
            reward_value REAL,
            reward_type TEXT,
            key TEXT,
            annotation TEXT,
            source TEXT,
            created_at TEXT
        );

        CREATE TABLE outcome_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            total_reward REAL,
            reward_metadata TEXT,
            created_at TEXT
        );
        """
    )


def _seed_minimal_session(conn: sqlite3.Connection, session_id: str = "s1") -> None:
    unique_reward_annotation = json.dumps({"new_unique": ["A"]})
    regular_reward_annotation = json.dumps({})
    outcome_reward_metadata = json.dumps({"achievements": ["A", "B"]})

    conn.execute(
        "INSERT INTO session_traces(session_id, created_at, metadata) VALUES (?, ?, ?)",
        (session_id, "2025-01-01T00:00:00Z", '{"episode_id": 123, "run_id": "r1"}'),
    )
    conn.execute(
        "INSERT INTO session_timesteps(session_id, step_id, step_index, turn_number, started_at, completed_at, step_metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, "st1", 0, 1, "t0", "t1", '{"info": 1}'),
    )
    conn.execute(
        "INSERT INTO events(session_id, event_type, system_instance_id, event_time, message_time, created_at, model_name, provider, input_tokens, output_tokens, total_tokens, cost_usd, latency_ms, span_id, trace_id, call_records, reward, terminated, truncated, system_state_before, system_state_after, metadata, event_metadata) VALUES (?, 'model_call', 'sys1', 't0', 't0', 't0', 'm', 'prov', 1, 1, 2, 0.0, 10.0, 'sp', 'tr', '[]', 0.0, 0, 0, '{}', '{}', '{}', '{}')",
        (session_id,),
    )
    conn.execute(
        "INSERT INTO messages(session_id, message_type, content, event_time, message_time, timestamp, metadata) VALUES (?, 'user', 'hello', 't0', 't0', 't0', '{}')",
        (session_id,),
    )
    # One unique achievement delta and one regular achievement
    conn.execute(
        """
        INSERT INTO event_rewards(session_id, event_id, turn_number, reward_value, reward_type, key, annotation, source, created_at)
        VALUES (?, 1, 1, 1.5, 'unique_achievement_delta', 'k', ?, 'test', 't0')
        """,
        (session_id, unique_reward_annotation),
    )
    conn.execute(
        """
        INSERT INTO event_rewards(session_id, event_id, turn_number, reward_value, reward_type, key, annotation, source, created_at)
        VALUES (?, 1, 1, 2.0, 'achievement_delta', 'k', ?, 'test', 't0')
        """,
        (session_id, regular_reward_annotation),
    )
    conn.execute(
        """
        INSERT INTO outcome_rewards(session_id, total_reward, reward_metadata, created_at)
        VALUES (?, 3.5, ?, 't0')
        """,
        (session_id, outcome_reward_metadata),
    )
    conn.commit()


def test_fetch_crafter_sessions_basic() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    _seed_minimal_session(conn, "s1")
    _seed_minimal_session(conn, "s2")

    session_ids = fetch_crafter_sessions(conn, limit=10, metadata_filter=None, session_ids=None, min_event_count=0)
    assert set(session_ids) == {"s1", "s2"}

    filtered = fetch_crafter_sessions(conn, limit=10, metadata_filter="episode", session_ids=None, min_event_count=0)
    assert set(filtered) == {"s1", "s2"}


def test_load_session_trace_and_metrics() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    _seed_minimal_session(conn, "s1")

    trace = load_session_trace(conn, "s1")
    assert trace["session_id"] == "s1"
    assert isinstance(trace["event_history"], list) and len(trace["event_history"]) == 1
    assert isinstance(trace["markov_blanket_message_history"], list) and len(trace["markov_blanket_message_history"]) == 1

    metrics = compute_deterministic_metrics(conn, "s1")
    assert isinstance(metrics, DeterministicMetrics)
    assert metrics.unique_achievement_reward == pytest.approx(1.5)
    assert metrics.achievement_reward == pytest.approx(2.0)
    assert metrics.outcome_total_reward == pytest.approx(3.5)
    assert metrics.unique_achievement_count == 1
    assert metrics.final_achievement_count == 2

