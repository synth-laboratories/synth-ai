from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from synth_ai.tracing_v3.trace_utils import load_session_trace, compute_deterministic_metrics
from synth_ai.judge_schemas import JudgeScoreResponse


async def _process_trace(
    *,
    client: Any,
    conn: Any,
    session_id: str,
    args: Any,
    options: dict[str, Any],
    output_dir: Path,
    rubric_cfg: dict[str, Any],
):
    trace = load_session_trace(conn, session_id)
    deterministic = compute_deterministic_metrics(conn, session_id)

    # Build judge trace
    event_history = trace.get("event_history") or trace.get("events") or []
    if not isinstance(event_history, list):
        event_history = []
    event_history = [item for item in event_history if isinstance(item, dict)]
    if not event_history:
        raise ValueError("Trace missing event_history entries")

    markov_history = trace.get("markov_blanket_message_history") or trace.get("messages") or []
    if not isinstance(markov_history, list):
        markov_history = []
    markov_history = [item for item in markov_history if isinstance(item, dict)]

    metadata = trace.get("metadata")
    metadata_dict: dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}

    session_id_meta = trace.get("session_id")
    if session_id_meta and isinstance(session_id_meta, str):
        metadata_dict.setdefault("session_id", session_id_meta)
        metadata_dict.setdefault("trace_id", session_id_meta)

    payload = {
        "policy_name": args.policy_name,
        "task_app": {"id": args.task_app_id},
        "trace": {
            "event_history": event_history,
            "markov_blanket_message_history": markov_history,
            "metadata": metadata_dict,
        },
        "options": options,
    }

    resp = await client.post(args.backend_url.rstrip("/") + "/api/judge/v1/score", json=payload, headers={})
    response = resp.json()
    validated = JudgeScoreResponse.model_validate(response)
    judge_event = validated.aggregate_event_reward()
    judge_outcome = validated.aggregate_outcome_reward()

    # Write outputs
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_json(output_dir / "traces" / f"{session_id}.json", trace)
    _save_json(
        output_dir / "summaries" / f"{session_id}.json",
        {
            "session_id": session_id,
            "deterministic_event_reward": deterministic.unique_achievement_reward,
            "deterministic_outcome_reward": deterministic.outcome_total_reward,
            "judge_event_reward": judge_event,
            "judge_outcome_reward": judge_outcome,
            "rubric_config": rubric_cfg,
            "judge_options": options,
        },
    )
    _save_json(output_dir / "judgements" / f"{session_id}.json", response)

    return (
        {
            "session_id": session_id,
            "deterministic_event_reward": deterministic.unique_achievement_reward,
            "deterministic_outcome_reward": deterministic.outcome_total_reward,
            "judge_event_reward": judge_event,
            "judge_outcome_reward": judge_outcome,
        },
        0.0,
        {"total_duration_ms": 0.0, "num_events": len(event_history)},
    )


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class DummyClient:
    async def post(self, url: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> DummyResponse:  # type: ignore[override]
        # Return a minimal, valid JudgeScoreResponse payload
        return DummyResponse(
            {
                "status": "ok",
                "event_reviews": [{"criteria": {}, "total": 0.0}],
                "outcome_review": {"criteria": {}, "total": 1.25},
                "event_totals": [0.5],
                "details": {"provider_latency_ms": 10.0},
                "metadata": {"step_latency": {"mean_ms": 5.0, "count": 1}},
            }
        )


def _make_in_memory_conn():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Minimal schema to satisfy trace_utils.load_session_trace()
    conn.executescript(
        """
        CREATE TABLE session_traces (session_id TEXT PRIMARY KEY, created_at TEXT, metadata TEXT);
        CREATE TABLE session_timesteps (
            session_id TEXT, step_id TEXT, step_index INTEGER, turn_number INTEGER,
            started_at TEXT, completed_at TEXT, step_metadata TEXT
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_type TEXT,
            system_instance_id TEXT, event_time TEXT, message_time TEXT, created_at TEXT,
            model_name TEXT, provider TEXT, input_tokens INTEGER, output_tokens INTEGER,
            total_tokens INTEGER, cost_usd REAL, latency_ms REAL, span_id TEXT, trace_id TEXT,
            call_records TEXT, reward REAL, terminated INTEGER, truncated INTEGER,
            system_state_before TEXT, system_state_after TEXT, metadata TEXT, event_metadata TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_type TEXT, content TEXT,
            event_time TEXT, message_time TEXT, timestamp TEXT, metadata TEXT
        );
        CREATE TABLE event_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_id INTEGER, turn_number INTEGER,
            reward_value REAL, reward_type TEXT, key TEXT, annotation TEXT, source TEXT, created_at TEXT
        );
        CREATE TABLE outcome_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, total_reward REAL, reward_metadata TEXT, created_at TEXT
        );
        """
    )

    # Seed minimal data for a single session
    conn.execute(
        "INSERT INTO session_traces(session_id, created_at, metadata) VALUES ('s1', 't0', '{"episode_id": 1}')"
    )
    conn.execute(
        "INSERT INTO session_timesteps(session_id, step_id, step_index, turn_number, started_at, completed_at, step_metadata) VALUES ('s1', 'st1', 0, 1, 't0', 't1', '{}')"
    )
    conn.execute(
        "INSERT INTO events(session_id, event_type, system_instance_id, event_time, message_time, created_at, model_name, provider, input_tokens, output_tokens, total_tokens, cost_usd, latency_ms, span_id, trace_id, call_records, reward, terminated, truncated, system_state_before, system_state_after, metadata, event_metadata) VALUES ('s1', 'model_call', 'sys', 't0', 't0', 't0', 'm', 'prov', 1, 1, 2, 0.0, 1.0, 'sp', 'tr', '[]', 0.0, 0, 0, '{}', '{}', '{}', '{}')"
    )
    conn.execute(
        "INSERT INTO messages(session_id, message_type, content, event_time, message_time, timestamp, metadata) VALUES ('s1', 'user', 'hello', 't0', 't0', 't0', '{}')"
    )
    conn.execute(
        "INSERT INTO event_rewards(session_id, event_id, turn_number, reward_value, reward_type, key, annotation, source, created_at) VALUES ('s1', 1, 1, 1.0, 'unique_achievement_delta', 'k', '{"new_unique": ["X"]}', 't', 't0')"
    )
    conn.execute(
        "INSERT INTO outcome_rewards(session_id, total_reward, reward_metadata, created_at) VALUES ('s1', 2.0, '{"achievements": ["Y"]}', 't0')"
    )
    conn.commit()

    return conn


@pytest.mark.asyncio
async def test_process_trace_happy_path(tmp_path: Path) -> None:
    client = DummyClient()
    conn = _make_in_memory_conn()

    class Args:
        backend_url = "http://test"
        api_key = None
        policy_name = "p"
        task_app_id = "id"
        task_app_base_url = None

    options: dict[str, Any] = {}
    out_dir = tmp_path / "rubrics_out"
    rubric_cfg: dict[str, Any] = {"id": "r"}

    summary, dur_ms, diag = await _process_trace(
        client=client, conn=conn, session_id="s1", args=Args, options=options, output_dir=out_dir, rubric_cfg=rubric_cfg
    )

    assert summary is not None and dur_ms is not None and diag is not None
    assert summary["session_id"] == "s1"
    assert "deterministic_event_reward" in summary
    assert "judge_outcome_reward" in summary
    assert (out_dir / "traces" / "s1.json").exists()
    assert (out_dir / "judgements" / "s1.json").exists()
    assert (out_dir / "summaries" / "s1.json").exists()


