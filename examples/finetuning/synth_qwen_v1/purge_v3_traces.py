#!/usr/bin/env python3
"""
Purge v3 trace databases:
- Find all paths matching **/traces_v3_lm_synth/traces.db under the repo
- If the DB is inside an `old/` path → delete the DB (and -wal/-shm) outright
- Else → delete records older than 24 hours and VACUUM to reclaim space

Run with: uvpm examples.finetuning.synth_qwen.purge_v3_traces
"""

import contextlib
import datetime
import os
import shutil
import sqlite3
from pathlib import Path


def find_trace_dbs(repo_root: Path) -> list[Path]:
    return list(repo_root.rglob("traces_v3_lm_synth/traces.db"))


def delete_db_files(db_path: Path) -> None:
    wal = db_path.with_suffix(".db-wal")
    shm = db_path.with_suffix(".db-shm")
    if db_path.exists():
        os.remove(db_path)
    if wal.exists():
        os.remove(wal)
    if shm.exists():
        os.remove(shm)


def purge_older_than_24h(db_path: Path) -> None:
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=24)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Collect session_ids to purge
    cur.execute("SELECT session_id FROM session_traces WHERE created_at < ?", (cutoff,))
    session_ids = [row[0] for row in cur.fetchall()]

    if session_ids:
        placeholders = ",".join(["?"] * len(session_ids))
        cur.execute(f"DELETE FROM events WHERE session_id IN ({placeholders})", session_ids)
        cur.execute(f"DELETE FROM messages WHERE session_id IN ({placeholders})", session_ids)
        cur.execute(
            f"DELETE FROM session_timesteps WHERE session_id IN ({placeholders})", session_ids
        )
        cur.execute(
            f"DELETE FROM session_traces WHERE session_id IN ({placeholders}) AND created_at < ?",
            session_ids + [cutoff],
        )

    # Commit deletions before VACUUM
    con.commit()
    con.close()

    # Attempt VACUUM
    try:
        con2 = sqlite3.connect(str(db_path))
        cur2 = con2.cursor()
        cur2.execute("VACUUM")
        con2.commit()
        con2.close()
        return
    except sqlite3.OperationalError:
        with contextlib.suppress(Exception):
            con2.close()

    # Fallback: VACUUM INTO a temp path (e.g., /tmp) then replace atomically
    tmp_target = Path("/tmp") / f"{db_path.stem}_compacted.db"
    try:
        con3 = sqlite3.connect(str(db_path))
        cur3 = con3.cursor()
        cur3.execute(f"VACUUM INTO '{tmp_target.as_posix()}'")
        con3.commit()
        con3.close()

        # Replace original DB with compacted copy
        delete_db_files(db_path)
        shutil.move(str(tmp_target), str(db_path))
    finally:
        if tmp_target.exists():
            # Clean up if move failed
            with contextlib.suppress(Exception):
                os.remove(tmp_target)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dbs = find_trace_dbs(repo_root)
    print(f"🔎 Found {len(dbs)} v3 trace DB(s)")

    for db in dbs:
        db_str = str(db)
        if "/old/" in db_str or db_str.endswith("/old/traces_v3_lm_synth/traces.db"):
            print(f"🗑️  Deleting DB under old/: {db_str}")
            delete_db_files(db)
            continue
        print(f"🧹 Purging records older than 24h: {db_str}")
        purge_older_than_24h(db)
        print(f"✅ Purged and compacted: {db_str}")


if __name__ == "__main__":
    main()
