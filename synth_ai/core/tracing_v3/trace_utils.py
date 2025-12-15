from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Row = sqlite3.Row


def connect(db_path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _json_load(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return value
    if isinstance(value, bytes | bytearray):
        value = value.decode("utf-8", errors="ignore")
    try:
        return json.loads(value)
    except Exception:
        return value


def _row_get(row: Row, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        return None


def fetch_crafter_sessions(
    conn: sqlite3.Connection,
    *,
    limit: int,
    metadata_filter: str | None = None,
    session_ids: Sequence[str] | None = None,
    min_event_count: int = 0,
) -> list[str]:
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        rows = conn.execute(
            f"""
            SELECT session_id
            FROM session_traces
            WHERE session_id IN ({placeholders})
            ORDER BY created_at DESC
            """,
            tuple(session_ids),
        ).fetchall()
        return [row["session_id"] for row in rows]

    params: list[Any] = []
    where_clauses: list[str] = []
    if metadata_filter:
        where_clauses.append("session_traces.metadata LIKE ?")
        params.append(f"%{metadata_filter}%")
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    having_sql = ""
    if min_event_count > 0:
        having_sql = "HAVING COUNT(events.id) >= ?"
        params.append(min_event_count)

    query = f"""
        SELECT session_traces.session_id
        FROM session_traces
        LEFT JOIN events ON session_traces.session_id = events.session_id
        {where_sql}
        GROUP BY session_traces.session_id
        {having_sql}
        ORDER BY session_traces.created_at DESC
        LIMIT ?
    """
    rows = conn.execute(query, (*params, limit)).fetchall()
    return [row["session_id"] for row in rows]


def load_session_trace(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    session_row = conn.execute(
        """
        SELECT session_id, created_at, metadata
        FROM session_traces
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if not session_row:
        raise ValueError(f"Session {session_id} not found")

    timesteps = conn.execute(
        """
        SELECT step_id,
               step_index,
               turn_number,
               started_at,
               completed_at,
               step_metadata
        FROM session_timesteps
        WHERE session_id = ?
        ORDER BY step_index ASC
        """,
        (session_id,),
    ).fetchall()

    event_rows = conn.execute(
        """
        SELECT *
        FROM events
        WHERE session_id = ?
        ORDER BY event_time ASC, id ASC
        """,
        (session_id,),
    ).fetchall()

    message_rows = conn.execute(
        """
        SELECT *
        FROM messages
        WHERE session_id = ?
        ORDER BY event_time ASC, id ASC
        """,
        (session_id,),
    ).fetchall()

    event_rewards = conn.execute(
        """
        SELECT *
        FROM event_rewards
        WHERE session_id = ?
        ORDER BY turn_number ASC, id ASC
        """,
        (session_id,),
    ).fetchall()

    outcome_rewards = conn.execute(
        """
        SELECT *
        FROM outcome_rewards
        WHERE session_id = ?
        ORDER BY created_at ASC
        """,
        (session_id,),
    ).fetchall()

    metadata = _json_load(session_row["metadata"]) or {}
    if isinstance(metadata, dict):
        episode_id = metadata.get("episode_id")
        if episode_id is not None and not isinstance(episode_id, str):
            metadata["episode_id"] = str(episode_id)

    events_payload = [
        {
            "id": row["id"],
            "event_type": row["event_type"],
            "system_instance_id": row["system_instance_id"],
            "time_record": {
                "event_time": row["event_time"],
                "message_time": row["message_time"],
                "created_at": row["created_at"],
            },
            "model_name": row["model_name"],
            "provider": row["provider"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "cost_usd": row["cost_usd"],
            "latency_ms": row["latency_ms"],
            "span_id": row["span_id"],
            "trace_id": row["trace_id"],
            "call_records": _json_load(row["call_records"]) or [],
            "reward": row["reward"],
            "terminated": row["terminated"],
            "truncated": row["truncated"],
            "system_state_before": _json_load(row["system_state_before"]),
            "system_state_after": _json_load(row["system_state_after"]),
            "metadata": _json_load(row["metadata"]) or {},
            "event_metadata": _json_load(row["event_metadata"]),
        }
        for row in event_rows
    ]

    messages_payload = [
        {
            "id": row["id"],
            "message_type": row["message_type"],
            "content": row["content"],
            "time_record": {
                "event_time": row["event_time"],
                "message_time": row["message_time"],
                "timestamp": row["timestamp"],
            },
            "metadata": _json_load(row["metadata"]) or {},
        }
        for row in message_rows
    ]

    trace: dict[str, Any] = {
        "session_id": session_row["session_id"],
        "created_at": session_row["created_at"],
        "metadata": metadata,
        "session_time_steps": [
            {
                "step_id": row["step_id"],
                "step_index": row["step_index"],
                "turn_number": row["turn_number"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "metadata": _json_load(row["step_metadata"]) or {},
            }
            for row in timesteps
        ],
        "event_history": events_payload,
        "events": events_payload,
        "markov_blanket_message_history": messages_payload,
        "messages": messages_payload,
        "event_rewards": [
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "turn_number": row["turn_number"],
                "reward_value": row["reward_value"],
                "reward_type": row["reward_type"],
                "key": row["key"],
                "annotation": _json_load(row["annotation"]) or {},
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in event_rewards
        ],
        "outcome_rewards": [
            {
                "id": row["id"],
                "total_reward": row["total_reward"],
                "achievements_count": row["achievements_count"],
                "total_steps": row["total_steps"],
                "reward_metadata": _json_load(row["reward_metadata"]) or {},
                "annotation": _json_load(_row_get(row, "annotation")) or {},
                "created_at": row["created_at"],
            }
            for row in outcome_rewards
        ],
    }
    return trace


@dataclass
class DeterministicMetrics:
    session_id: str
    unique_achievement_reward: float
    achievement_reward: float
    outcome_total_reward: float
    unique_achievement_count: int
    final_achievement_count: int


def compute_deterministic_metrics(conn: sqlite3.Connection, session_id: str) -> DeterministicMetrics:
    event_rows = conn.execute(
        """
        SELECT reward_type, reward_value, annotation
        FROM event_rewards
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()

    unique_total = 0.0
    all_total = 0.0
    unique_achievements: set[str] = set()

    for row in event_rows:
        reward_type = row["reward_type"]
        value = float(row["reward_value"] or 0.0)
        if reward_type == "unique_achievement_delta":
            unique_total += value
            annotation = _json_load(row["annotation"]) or {}
            for name in annotation.get("new_unique") or []:
                if isinstance(name, str):
                    unique_achievements.add(name)
        elif reward_type == "achievement_delta":
            all_total += value

    outcome_rows = conn.execute(
        """
        SELECT total_reward, reward_metadata
        FROM outcome_rewards
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()

    outcome_total = 0.0
    final_achievements: set[str] = set()
    for row in outcome_rows:
        outcome_total += float(row["total_reward"] or 0.0)
        metadata = _json_load(row["reward_metadata"]) or {}
        for name in metadata.get("achievements") or []:
            if isinstance(name, str):
                final_achievements.add(name)

    return DeterministicMetrics(
        session_id=session_id,
        unique_achievement_reward=unique_total,
        achievement_reward=all_total,
        outcome_total_reward=outcome_total,
        unique_achievement_count=len(unique_achievements),
        final_achievement_count=len(final_achievements),
    )


__all__ = [
    "DeterministicMetrics",
    "compute_deterministic_metrics",
    "connect",
    "fetch_crafter_sessions",
    "load_session_trace",
]
