import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def _safe_chmod(path: str | Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError as e:
        logger.warning("Failed to set permissions %o on %s: %s", mode, path, e)


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _safe_chmod(path, PRIVATE_DIR_MODE)


def write_private_text(path: Path, content: str, *, mode: int = PRIVATE_FILE_MODE) -> None:
    ensure_private_dir(path.parent)
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        tmp_path = None
        _safe_chmod(path, mode)
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def write_private_json(
    path: Path,
    data: Mapping[str, Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    payload = json.dumps(dict(data), indent=indent, sort_keys=sort_keys) + "\n"
    write_private_text(path, payload)
