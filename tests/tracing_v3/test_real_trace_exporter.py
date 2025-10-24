#!/usr/bin/env python3
"""
Regression tests for the trace SFT exporter using real fixtures.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from examples.warming_up_to_rl import export_trace_sft as exporter


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces"


def _fixture_db(scenario: str) -> Path:
    db_path = FIXTURE_ROOT / scenario / "synth_ai.db"
    if not db_path.exists():
        raise RuntimeError(f"Fixture database missing for scenario '{scenario}': {db_path}")
    return db_path


@pytest.mark.fast
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
