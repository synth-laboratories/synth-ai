import sqlite3
import threading
import contextlib
from pathlib import Path
from typing import Optional


class SQLiteManager:
    def __init__(self, db_path: Path, read_only: bool = True, ephemeral: bool = False):
        """Initializes SQLiteManager with optional read-only or ephemeral mode."""
        self.db_path = db_path
        self.read_only = read_only
        self.ephemeral = ephemeral
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    @contextlib.contextmanager
    def connection(self):
        with self._lock:
            if self._conn is None:
                uri = f"file:{self.db_path}?mode=ro" if self.read_only else str(self.db_path)
                self._conn = sqlite3.connect(uri, uri=self.read_only, isolation_level="DEFERRED")
                self._conn.execute("PRAGMA foreign_keys=ON;")
                self._conn.execute("PRAGMA journal_mode=WAL;")
            try:
                yield self._conn
                self._conn.commit()
            except:
                self._conn.rollback()
                raise

    def close(self):
        """Closes the underlying connection for cleanup at environment teardown."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def reset(self, new_db_path: Optional[Path] = None):
        """Resets the connection, optionally switching to a new database path for a fresh session."""
        with self._lock:
            if self._conn:
                self._conn.close()
            if new_db_path:
                self.db_path = new_db_path
            self._conn = None
