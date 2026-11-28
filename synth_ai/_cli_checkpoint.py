"""Per-run event stream checkpoint persistence for `synth logs --follow`."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from ._stream import StreamCheckpoint, StreamCheckpointData

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def state_dir() -> Path:
    configured = os.getenv("SYNTH_STATE_DIR")
    return (
        Path(configured).expanduser()
        if configured is not None and configured != ""
        else Path.home() / ".config" / "synth"
    )


def checkpoint_path(run_id: str) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run id cannot name a checkpoint file")
    return state_dir() / "streams" / (run_id + ".json")


def load_checkpoint(run_id: str) -> StreamCheckpoint:
    path = checkpoint_path(run_id)
    if not path.exists():
        return StreamCheckpoint(run_id=run_id)
    if path.stat().st_mode & 0o077:
        raise ValueError("checkpoint file must have permissions 0600")
    checkpoint = StreamCheckpoint.from_data(
        StreamCheckpointData.model_validate_json(path.read_bytes())
    )
    checkpoint.bind(run_id)
    return checkpoint


def save_checkpoint(checkpoint: StreamCheckpoint) -> None:
    if checkpoint.run_id is None:
        raise ValueError("checkpoint is not bound to a run")
    path = checkpoint_path(checkpoint.run_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(checkpoint.to_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
