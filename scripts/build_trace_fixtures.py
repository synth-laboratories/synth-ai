#!/usr/bin/env python3
"""Regenerate trace fixtures used by regression tests.

This script extracts representative subsets from a full tracing_v3 database
and writes compact fixture databases under ``tests/artifacts/traces``.

Current scenarios:
    - ``chat_small``: first three sessions chronologically
    - ``env_rollout``: two sessions with the highest number of environment events
    - ``high_volume``: single session with the highest total events
"""



import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import shutil
import hashlib

from synth_ai.tracing_v3.constants import (
    TRACE_DB_BASENAME,
    TRACE_DB_DIR,
    canonical_trace_db_name,
)

DEFAULT_TABLES: Sequence[str] = (
    "session_traces",
    "session_timesteps",
    "events",
    "messages",
    "event_rewards",
    "outcome_rewards",
)


@dataclass
class ScenarioSpec:
    name: str
    description: str
    session_ids: list[str]


def _copy_schema(src: sqlite3.Connection, dst: sqlite3.Connection, tables: Iterable[str]) -> None:
    for table in tables:
        row = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row and row["sql"]:
            dst.execute(row["sql"])
    dst.commit()


def _copy_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    where_clause: str,
    params: Sequence,
) -> int:
    pragma_cols = src.execute(f"PRAGMA table_info({table})").fetchall()
    if not pragma_cols:
        return 0
    columns = [col["name"] for col in pragma_cols]
    query = f"SELECT {', '.join(columns)} FROM {table} WHERE {where_clause}"
    rows = src.execute(query, params).fetchall()
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    dst.executemany(insert_sql, ([row[col] for col in columns] for row in rows))
    return len(rows)


def _write_manifest(
    dst_path: Path,
    spec: ScenarioSpec,
    counts: Counter,
    extra: dict[str, object],
) -> None:
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        if path.exists():
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        return h.hexdigest()

    manifest = {
        "description": spec.description,
        "source": str(extra.pop("source_db")),
        "session_ids": spec.session_ids,
        **extra,
        "counts": dict(counts),
        "hashes": {
            canonical_trace_db_name(): _hash_file(dst_path / canonical_trace_db_name()),
            "trace_export.jsonl": _hash_file(dst_path / "trace_export.jsonl"),
        },
    }
    (dst_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_trace_export(dst_conn: sqlite3.Connection, dst_dir: Path) -> None:
    rows = dst_conn.execute(
        """
        SELECT session_id, created_at, num_timesteps, num_events, num_messages
        FROM session_traces
        ORDER BY created_at
        """
    ).fetchall()
    with (dst_dir / "trace_export.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in row.keys()}, default=str) + "\n")


def _build_chat_small(conn: sqlite3.Connection) -> ScenarioSpec:
    rows = conn.execute(
        """
        SELECT session_id
        FROM session_traces
        ORDER BY created_at ASC
        LIMIT 3
        """
    ).fetchall()
    return ScenarioSpec(
        name="chat_small",
        description="Subset of three early sessions extracted from local traces/v3 for regression tests.",
        session_ids=[row["session_id"] for row in rows],
    )


def _build_env_rollout(conn: sqlite3.Connection) -> ScenarioSpec:
    rows = conn.execute(
        """
        SELECT session_id, COUNT(*) AS c
        FROM events
        WHERE event_type='environment'
        GROUP BY session_id
        ORDER BY c DESC
        LIMIT 2
        """
    ).fetchall()
    return ScenarioSpec(
        name="env_rollout",
        description="Two sessions rich in environment events for rollout validation.",
        session_ids=[row["session_id"] for row in rows],
    )


def _build_high_volume(conn: sqlite3.Connection) -> ScenarioSpec:
    row = conn.execute(
        """
        SELECT session_id
        FROM session_traces
        ORDER BY num_events DESC
        LIMIT 1
        """
    ).fetchone()
    return ScenarioSpec(
        name="high_volume",
        description="Single session with the highest number of events to stress high-throughput paths.",
        session_ids=[row["session_id"]],
    )


def _prepare_destination(root: Path, name: str, overwrite: bool) -> Path:
    target = root / name
    if target.exists() and overwrite:
        for child in target.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _extract_fixture(
    conn: sqlite3.Connection,
    spec: ScenarioSpec,
    dest_root: Path,
    overwrite: bool,
) -> None:
    dst_dir = _prepare_destination(dest_root, spec.name, overwrite)
    dst_db_path = dst_dir / canonical_trace_db_name()
    if dst_db_path.exists() and overwrite:
        dst_db_path.unlink()

    dst = sqlite3.connect(dst_db_path)
    dst.row_factory = sqlite3.Row
    try:
        _copy_schema(conn, dst, DEFAULT_TABLES)
        placeholders = ", ".join("?" for _ in spec.session_ids)
        where_session = f"session_id IN ({placeholders})"

        for table in DEFAULT_TABLES:
            _copy_rows(conn, dst, table, where_session, spec.session_ids)

        # Copy systems referenced by the selected sessions
        system_rows = conn.execute(
            f"""
            SELECT DISTINCT system_instance_id
            FROM events
            WHERE {where_session} AND system_instance_id IS NOT NULL
            """,
            spec.session_ids,
        ).fetchall()
        system_ids = [row["system_instance_id"] for row in system_rows if row["system_instance_id"]]
        if system_ids:
            _copy_schema(conn, dst, ["systems"])
            placeholders_sys = ", ".join("?" for _ in system_ids)
            _copy_rows(conn, dst, "systems", f"system_id IN ({placeholders_sys})", system_ids)

        dst.commit()

        counts = Counter()
        for table in DEFAULT_TABLES:
            counts[table] = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if dst.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='systems'"
        ).fetchone():
            counts["systems"] = dst.execute("SELECT COUNT(*) FROM systems").fetchone()[0]

        _write_trace_export(dst, dst_dir)
        extra = {"source_db": conn.execute("PRAGMA database_list").fetchone()["file"]}
        _write_manifest(dst_dir, spec, counts, extra)
    finally:
        dst.close()


def _discover_source_db() -> Path:
    """Find the most recent task app trace database under the default trace directory."""

    candidates: list[Path] = []
    if TRACE_DB_DIR.exists():
        for path in TRACE_DB_DIR.glob(f"{TRACE_DB_BASENAME}_*.db"):
            candidates.append(path)
        fallback = TRACE_DB_DIR / canonical_trace_db_name()
        if fallback.exists():
            candidates.append(fallback)
    if not candidates:
        raise SystemExit(
            "Trace database not found. Provide --source or generate a task app trace first."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to the full tracing_v3 SQLite database (default: latest task_app_traces_*.db).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("tests/artifacts/traces"),
        help="Directory where fixtures will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing fixture files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source or _discover_source_db()
    if not source.exists():
        raise SystemExit(f"Source database not found: {source}")

    args.dest.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    try:
        builders = (_build_chat_small, _build_env_rollout, _build_high_volume)
        for builder in builders:
            spec = builder(conn)
            if not spec.session_ids:
                raise RuntimeError(f"No sessions found for scenario '{spec.name}'")
            _extract_fixture(conn, spec, args.dest, overwrite=args.overwrite)
            print(f"Regenerated fixture: {spec.name} ({len(spec.session_ids)} session(s))")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
