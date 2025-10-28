#!/usr/bin/env python3
"""
ReplicaSync regression tests using real fixture databases.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from synth_ai.tracing_v3.replica_sync import ReplicaSync
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


@pytest.mark.asyncio
@pytest.mark.fast
async def test_replica_sync_uses_fixture_database(monkeypatch, tmp_path):
    """ReplicaSync should target the provided fixture path and invoke libsql sync."""
    source = _fixture_db("env_rollout")
    local_db = tmp_path / "embedded.db"
    shutil.copy2(source, local_db)

    captured = {}

    class FakeConn:
        def __init__(self, path: str, sync_url: str | None, auth_token: str | None, sync_interval: int):
            captured["path"] = Path(path)
            captured["sync_url"] = sync_url
            captured["auth_token"] = auth_token
            captured["sync_interval"] = sync_interval
            self.sync_called = False

        def sync(self):
            self.sync_called = True
            captured["sync_called"] = True

        def close(self):
            captured["closed"] = True

    async def fake_to_thread(func, *args, **kwargs):
        func(*args, **kwargs)

    monkeypatch.setattr(
        "synth_ai.tracing_v3.replica_sync.libsql.connect",
        lambda *args, **kwargs: FakeConn(*args, **kwargs),
    )
    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    syncer = ReplicaSync(
        db_path=str(local_db),
        sync_url="libsql://fixture",
        auth_token="fixture-token",
        sync_interval=0,
    )

    success = await syncer.sync_once()
    await syncer.stop()

    assert success is True
    assert captured["path"] == local_db
    assert captured["sync_url"] == "libsql://fixture"
    assert captured["auth_token"] == "fixture-token"
    assert captured["sync_interval"] == 0
    assert captured.get("sync_called") is True
