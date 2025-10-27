import os
import shutil
from pathlib import Path

import pytest

from synth_ai.api.train.utils import validate_sft_jsonl
from synth_ai.tracing_v3.constants import TRACE_DB_BASENAME, canonical_trace_db_name

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _artifacts_dir() -> Path:
    return _repo_root() / "tests" / "artifacts"


@pytest.mark.fast
def test_export_from_cached_trace_db(tmp_path: Path) -> None:
    # Use the committed minimal SQL to create a tiny SQLite DB on the fly
    fixture_dir = _artifacts_dir() / "traces" / "chat_small"
    candidates = sorted(
        fixture_dir.glob(f"{TRACE_DB_BASENAME}*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(fixture_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Unable to locate fixture database in {fixture_dir}")
    source_db = candidates[0]

    db_path = tmp_path / "traces" / "v3" / canonical_trace_db_name()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, db_path)

    # Import exporter pieces from the example to avoid reimplementation
    from examples.warming_up_to_rl.export_trace_sft import (
        connect,
        fetch_achievement_data,
        fetch_session_models,
        fetch_outcome_rewards,
        fetch_event_reward_totals,
        build_sft_dataset,
        write_jsonl,
    )

    # Build dataset with minimal filters
    out_path = tmp_path / "datasets" / "crafter_reject_sft.small.jsonl"
    conn = connect(db_path)
    try:
        achievements_map, unique_counts, _, _, session_unique_sets, final_ach = (
            fetch_achievement_data(conn)
        )
        session_models = fetch_session_models(conn)
        _outcome = fetch_outcome_rewards(conn)
        _event_totals = fetch_event_reward_totals(conn)

        eligible_sessions = set(session_models.keys())
        dataset = build_sft_dataset(
            conn,
            achievements_map,
            eligible_sessions,
            allowed_models=None,
            limit=3,
        )
        assert dataset, "expected at least one exported example"
        write_jsonl(out_path, dataset)
    finally:
        conn.close()

    # Validate JSONL matches SFT schema
    validate_sft_jsonl(out_path)
