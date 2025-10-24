#!/usr/bin/env python3
"""
Regression tests for the trace SFT exporter using real fixtures.
"""

from __future__ import annotations

from pathlib import Path

from examples.warming_up_to_rl import export_trace_sft as exporter
from synth_ai.tracing_v3.constants import TRACE_DB_BASENAME


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces"


def _fixture_db(scenario: str) -> Path:
    scenario_dir = FIXTURE_ROOT / scenario
    candidates = sorted(
        scenario_dir.glob(f"{TRACE_DB_BASENAME}*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(scenario_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Fixture database missing for scenario '{scenario}' in {scenario_dir}")
    return candidates[0]


def test_exporter_builds_dataset_from_env_rollout_fixture():
    """Ensure build_sft_dataset produces validated records from the env_rollout fixture."""
    db_path = _fixture_db("env_rollout")
    conn = exporter.connect(db_path)
    try:
        (
            achievements_map,
            _unique_counts,
            _name_counts,
            _size_counts,
            _session_unique_sets,
            _session_final_achievements,
        ) = exporter.fetch_achievement_data(conn)
        session_models = exporter.fetch_session_models(conn)
        eligible_sessions = set(session_models.keys())
        assert eligible_sessions, "Fixture should contain at least one eligible session"

        dataset = exporter.build_sft_dataset(
            conn,
            achievements_map,
            eligible_sessions,
            allowed_models=None,
            limit=None,
        )
        assert dataset, "Exporter should yield at least one training example"

        exporter._validate_dataset(dataset)

        emitted_sessions = {
            record.get("metadata", {}).get("session_id")
            for record in dataset
            if isinstance(record, dict)
        }
        emitted_sessions.discard(None)
        assert emitted_sessions.issubset(eligible_sessions)
    finally:
        conn.close()
