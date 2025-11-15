from __future__ import annotations

import importlib

from sqlalchemy import text

from synth_ai.experiment_queue import config as queue_config
from synth_ai.experiment_queue import database as queue_db


def test_init_db_enables_wal(tmp_path, monkeypatch):
    """Ensure init_db enables WAL mode for SQLite connections."""
    db_path = tmp_path / "wal_test.db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_path))
    queue_config.load_config.cache_clear()
    importlib.reload(queue_db)
    queue_db.init_db()
    engine = queue_db.get_engine()
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert journal_mode and journal_mode.lower() == "wal"
