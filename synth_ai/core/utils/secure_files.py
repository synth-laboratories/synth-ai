from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> None:
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        os.chmod(resolved, PRIVATE_DIR_MODE)


def write_private_text(path: Path, content: str, *, mode: int = PRIVATE_FILE_MODE) -> None:
    resolved = Path(path).expanduser()
    ensure_private_dir(resolved.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(resolved, flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise
    with suppress(OSError):
        os.chmod(resolved, mode)


def write_private_json(
    path: Path,
    data: Mapping[str, Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    payload = json.dumps(dict(data), indent=indent, sort_keys=sort_keys) + "\n"
    write_private_text(path, payload)
