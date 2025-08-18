"""
NOTE - first pass was o3-generated. Mostly bc idrk what I 'want' from this yet ...
trajectory_tree_store.py
~~~~~~~~~~~~~~~~~~~~~~~~
A minimal search-tree wrapper that pairs

  • an *in-memory* NetworkX DiGraph (parent ⇢ children edges)
  • a *content-addressable* FilesystemSnapshotStore (heavy blobs)

so you can implement things like LATS / MCTS without bringing in the
big “backend.production” code-base.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx

# from filesystem_snapshot_store import FilesystemSnapshotStore  # ← your re-impl

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# lightweight metadata record                                                 #
# --------------------------------------------------------------------------- #
import hashlib
import logging
import os

log = logging.getLogger(__name__)

# Default directory for storing snapshots relative to some base path
# This could be configured via environment variables or settings later.
DEFAULT_SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_STORE_PATH", "/tmp/agent_snapshots"))


class FilesystemSnapshotStore:
    """
    Stores and retrieves environment state snapshots on the filesystem.

    Uses content-addressable storage: the key (ID) for a snapshot
    is the SHA-256 hash of its compressed content.
    """

    def __init__(self, base_dir: str | Path = DEFAULT_SNAPSHOT_DIR):
        self.base_dir = Path(base_dir)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Initialized snapshot store at: {self.base_dir}")
        except OSError as e:
            log.error(
                f"Failed to create snapshot directory {self.base_dir}: {e}",
                exc_info=True,
            )
            raise

    def _get_path(self, key: str) -> Path:
        """Constructs the full path for a given snapshot key."""
        # Maybe add subdirectories later for large numbers of files, e.g., key[:2]/key[2:]
        filename = f"{key}.snapshot.gz"
        return self.base_dir / filename

    def write(self, blob: bytes | dict[str, Any]) -> str:
        """
        Stores a snapshot blob (bytes or dict) and returns its SHA-256 key.

        • Dicts → pickle → gzip
        • Bytes already gzip-compressed (magic 0x1f 0x8b) are stored as-is
          to avoid double compression.
        """
        try:
            if isinstance(blob, dict):
                compressed_blob = gzip.compress(pickle.dumps(blob))
            elif isinstance(blob, bytes):
                # Skip re-compression if data is already gzipped
                compressed_blob = blob if blob[:2] == b"\x1f\x8b" else gzip.compress(blob)
            else:
                raise TypeError(f"Unsupported blob type for snapshot store: {type(blob)}")

            key = hashlib.sha256(compressed_blob).hexdigest()
            path = self._get_path(key)
            if not path.exists():
                path.write_bytes(compressed_blob)
            return key
        except Exception as e:
            log.error(f"Failed to write snapshot: {e}", exc_info=True)
            raise

    def read(self, key: str) -> bytes | None:
        """
        Retrieves the raw *compressed* snapshot bytes for a given key.

        Returns None if the key is not found.
        Deserialization (decompression, unpickling) is the responsibility
        of the caller (e.g., ReproducibleResource.from_snapshot).
        """
        filepath = self._get_path(key)
        if not filepath.exists():
            log.warning(f"Snapshot key not found: {key}")
            return None
        try:
            with open(filepath, "rb") as f:
                compressed_blob = f.read()
            return compressed_blob
        except OSError as e:
            log.error(f"Failed to read snapshot {key} from {filepath}: {e}", exc_info=True)
            return None  # Or re-raise? Returning None might be safer.

    def exists(self, key: str) -> bool:
        """Checks if a snapshot with the given key exists."""
        return self._get_path(key).exists()


# Global instance (optional, could use dependency injection)
# snapshot_store = FilesystemSnapshotStore()


class TrajectorySnapshot:
    """
    A *metadata* header for one node in the search tree.
    The heavy serialized-state bytes live only in the snapshot store.
    """

    __slots__ = (
        "snap_id",
        "parent_id",
        "depth",
        "action",
        "reward",
        "terminated",
        "info",
    )

    def __init__(
        self,
        snap_id: str,
        parent_id: str | None,
        depth: int,
        action: Any | None,
        reward: float = 0.0,
        terminated: bool = False,
        info: dict[str, Any] | None = None,
    ):
        self.snap_id = snap_id
        self.parent_id = parent_id
        self.depth = depth
        self.action = action
        self.reward = reward
        self.terminated = bool(terminated)
        self.info = info or {}

    # helpful for printing / debugging
    def __repr__(self) -> str:  # pragma: no cover
        a = json.dumps(self.action) if self.action is not None else "∅"
        return (
            f"TrajSnap(id={self.snap_id[:7]}…, depth={self.depth}, "
            f"action={a}, reward={self.reward}, term={self.terminated})"
        )


# --------------------------------------------------------------------------- #
# tree manager                                                                #
# --------------------------------------------------------------------------- #


class TrajectoryTreeStore:
    """
    ❑  Adds snapshots (root / children) and keeps the DAG in-memory
    ❑  Optionally mirrors headers to a tiny SQLite DB (so you can kill +
       resume a long search)
    ❑  Hands out raw snapshot **bytes**; decoding is up to the caller.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # construction                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __init__(
        self,
        snapshot_store: FilesystemSnapshotStore | None = None,
        *,
        db_path: Path | str | None = None,
    ):
        self.snap_store = snapshot_store or FilesystemSnapshotStore()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.db_path = Path(db_path).expanduser() if db_path else None
        if self.db_path:
            self._init_sqlite()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # public API                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # insertion -------------------------------------------------------------

    def add_root(self, snapshot_blob: bytes, *, info: dict[str, Any] | None = None) -> str:
        """Insert the very first node and return its content-hash key."""
        snap_id = self.snap_store.write(snapshot_blob)
        self._add_node(TrajectorySnapshot(snap_id, None, 0, None, 0.0, False, info))
        return snap_id

    def add_child(
        self,
        parent_id: str,
        snapshot_blob: bytes,
        *,
        action: Any,
        reward: float,
        terminated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> str:
        """Attach `snapshot_blob` as a child reached by `action` from *parent_id*."""
        if parent_id not in self.graph:
            raise KeyError(f"Parent snapshot {parent_id[:8]}… not in tree")
        depth = self.graph.nodes[parent_id]["meta"].depth + 1  # type: ignore[index]
        snap_id = self.snap_store.write(snapshot_blob)
        meta = TrajectorySnapshot(snap_id, parent_id, depth, action, reward, terminated, info)
        self._add_node(meta)  # records node + (maybe) SQLite
        self.graph.add_edge(parent_id, snap_id, action=action, reward=reward)  # NX edge attrs
        return snap_id

    # read-side helpers -----------------------------------------------------

    def get_children(self, snap_id: str) -> tuple[str, ...]:
        return tuple(self.graph.successors(snap_id))

    def get_parent(self, snap_id: str) -> str | None:
        preds = tuple(self.graph.predecessors(snap_id))
        return preds[0] if preds else None

    def is_leaf(self, snap_id: str) -> bool:
        return self.graph.out_degree(snap_id) == 0

    # simple enumerations useful for MCTS / LATS ---------------------------

    def iter_leaves(self) -> Iterable[str]:
        """Yield snapshot-ids that currently have no children."""
        return (n for n in self.graph.nodes if self.is_leaf(n))

    def path_to_root(self, snap_id: str) -> tuple[str, ...]:
        """Return (snap_id, …, root_id)"""
        path = [snap_id]
        while (p := self.get_parent(path[-1])) is not None:
            path.append(p)
        return tuple(path)

    def reconstruct_actions(self, snap_id: str) -> tuple[Any, ...]:
        """Return the sequence of *actions* from the root → `snap_id`."""
        actions = []
        for child, parent in zip(self.path_to_root(snap_id)[:-1], self.path_to_root(snap_id)[1:], strict=False):
            actions.append(self.graph.edges[parent, child]["action"])
        return tuple(reversed(actions))

    # snapshot access -------------------------------------------------------

    def load_snapshot_blob(self, snap_id: str) -> bytes:
        blob = self.snap_store.read(snap_id)
        if blob is None:
            raise FileNotFoundError(f"Snapshot {snap_id[:8]}… missing on disk")
        return blob

    def load_pickled_payload(self, snap_id: str) -> Any:
        """Decompress + unpickle whatever you stored under this id."""
        return pickle.loads(gzip.decompress(self.load_snapshot_blob(snap_id)))

    # mutation --------------------------------------------------------------

    def prune_subtree(self, root_id: str) -> None:
        """Remove *root_id* and all its descendants from the in-mem graph and DB."""
        doomed = list(nx.dfs_preorder_nodes(self.graph, root_id))
        self.graph.remove_nodes_from(doomed)
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("DELETE FROM nodes WHERE snap_id = ?;", ((n,) for n in doomed))
                conn.executemany(
                    "DELETE FROM edges WHERE parent_id = ? OR child_id = ?;",
                    ((n, n) for n in doomed),
                )
                conn.commit()

    def wipe(self) -> None:
        """Clear the *entire* tree (does **not** delete snapshot files)."""
        self.graph.clear()
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("DELETE FROM nodes; DELETE FROM edges;")
                conn.commit()

    # ------------------------------------------------------------------- #
    # internal helpers                                                     #
    # ------------------------------------------------------------------- #

    def _add_node(self, meta: TrajectorySnapshot) -> None:
        self.graph.add_node(meta.snap_id, meta=meta)
        if self.db_path:
            self._sqlite_insert(meta)

    # ------------------------------------------------------------------- #
    # tiny SQLite backing store (optional)                                 #
    # ------------------------------------------------------------------- #

    def _init_sqlite(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes(
                    snap_id   TEXT PRIMARY KEY,
                    parent_id TEXT,
                    depth     INTEGER,
                    action    TEXT,
                    reward    REAL,
                    terminated INTEGER,
                    info      TEXT
                );
                CREATE TABLE IF NOT EXISTS edges(
                    parent_id TEXT,
                    child_id  TEXT,
                    action    TEXT,
                    reward    REAL,
                    PRIMARY KEY(parent_id, child_id)
                );
                """
            )
            conn.commit()

    def _sqlite_insert(self, meta: TrajectorySnapshot) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO nodes
                   (snap_id, parent_id, depth, action, reward, terminated, info)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    meta.snap_id,
                    meta.parent_id,
                    meta.depth,
                    json.dumps(meta.action),
                    meta.reward,
                    int(meta.terminated),
                    json.dumps(meta.info),
                ),
            )
            if meta.parent_id:
                conn.execute(
                    """INSERT OR IGNORE INTO edges
                       (parent_id, child_id, action, reward)
                       VALUES (?,?,?,?)""",
                    (
                        meta.parent_id,
                        meta.snap_id,
                        json.dumps(meta.action),
                        meta.reward,
                    ),
                )
            conn.commit()
