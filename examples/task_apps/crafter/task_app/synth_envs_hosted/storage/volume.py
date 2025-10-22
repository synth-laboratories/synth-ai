from __future__ import annotations

import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


class VolumeStorage:
    """Helpers for Modal Volume storage operations."""

    def __init__(self, base_path: str = "/data/state") -> None:
        self.base_path = Path(base_path)

    def get_snapshot_path(
        self,
        rl_run_id: str,
        kind: str,
        snapshot_id: str,
    ) -> Path:
        """Build the path for a snapshot file."""
        # Use first 2 chars of snapshot_id for sharding
        shard1 = snapshot_id[:2] if len(snapshot_id) >= 2 else "00"
        shard2 = snapshot_id[2:4] if len(snapshot_id) >= 4 else "00"

        return (
            self.base_path / "runs" / rl_run_id / kind / shard1 / shard2 / f"{snapshot_id}.tar.gz"
        )

    def get_index_path(self, rl_run_id: str) -> Path:
        """Get the index file path for a run."""
        return self.base_path / "runs" / rl_run_id / "index" / "meta.jsonl"

    def write_snapshot_atomic(
        self,
        path: Path,
        archive_bytes: bytes,
    ) -> None:
        """Atomically write a snapshot archive to disk."""
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            f.write(archive_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(tmp_path, path)

    def create_archive(
        self,
        state_dict: dict[str, Any],
        meta: dict[str, Any],
    ) -> bytes:
        """Create a tar.gz archive with state and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write state.json
            state_path = tmppath / "state.json"
            with open(state_path, "w") as f:
                json.dump(state_dict, f, sort_keys=True, indent=2)

            # Write meta.json
            meta_path = tmppath / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, sort_keys=True, indent=2)

            # Create tar archive
            tar_path = tmppath / "archive.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(state_path, arcname="state.json")
                tar.add(meta_path, arcname="meta.json")

            # Compress with gzip
            with open(tar_path, "rb") as f:
                tar_bytes = f.read()

            compressed = gzip.compress(tar_bytes, compresslevel=6)

            return compressed

    def extract_archive(self, archive_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract state and metadata from a tar.gz archive."""
        # Decompress
        tar_bytes = gzip.decompress(archive_bytes)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write tar bytes to temp file
            tar_path = tmppath / "archive.tar"
            with open(tar_path, "wb") as f:
                f.write(tar_bytes)

            # Extract tar
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(tmppath)

            # Read state and meta
            with open(tmppath / "state.json") as f:
                state = json.load(f)

            with open(tmppath / "meta.json") as f:
                meta = json.load(f)

            return state, meta

    def compute_snapshot_id(self, archive_bytes: bytes) -> str:
        """Compute content-addressed snapshot ID."""
        return hashlib.sha256(archive_bytes).hexdigest()

    def save_snapshot(
        self,
        rl_run_id: str,
        kind: str,
        state_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
        parent_snapshot_id: str | None = None,
    ) -> tuple[str, str, int]:
        """Save a snapshot and return (snapshot_id, path, size)."""
        # Build metadata
        meta = {
            "kind": kind,
            "rl_run_id": rl_run_id,
            "schema_version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
        }

        if parent_snapshot_id:
            meta["parent_snapshot_id"] = parent_snapshot_id

        if config:
            config_str = json.dumps(config, sort_keys=True)
            meta["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()

        # Create archive
        archive_bytes = self.create_archive(state_dict, meta)

        # Compute snapshot ID
        snapshot_id = self.compute_snapshot_id(archive_bytes)
        meta["snapshot_id"] = snapshot_id

        # Recreate archive with snapshot_id in metadata
        archive_bytes = self.create_archive(state_dict, meta)

        # Get path and write
        path = self.get_snapshot_path(rl_run_id, kind, snapshot_id)
        self.write_snapshot_atomic(path, archive_bytes)

        # Append to index
        self.append_to_index(rl_run_id, meta)

        return snapshot_id, str(path), len(archive_bytes)

    def load_snapshot(
        self,
        rl_run_id: str,
        kind: str,
        snapshot_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load a snapshot and return (state_dict, meta)."""
        path = self.get_snapshot_path(rl_run_id, kind, snapshot_id)

        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        with open(path, "rb") as f:
            archive_bytes = f.read()

        state, meta = self.extract_archive(archive_bytes)
        return state, meta

    def append_to_index(
        self,
        rl_run_id: str,
        meta: dict[str, Any],
    ) -> None:
        """Append metadata to the run's index file."""
        index_path = self.get_index_path(rl_run_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

    def read_index(self, rl_run_id: str) -> list[dict[str, Any]]:
        """Read all entries from a run's index file."""
        index_path = self.get_index_path(rl_run_id)

        if not index_path.exists():
            return []

        entries = []
        with open(index_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        return entries


# Global storage instance
storage = VolumeStorage()
