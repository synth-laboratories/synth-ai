from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.secure_files.") from exc

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> None:
    synth_ai_py.ensure_private_dir(str(path))


def write_private_text(path: Path, content: str, *, mode: int = PRIVATE_FILE_MODE) -> None:
    synth_ai_py.write_private_text(str(path), content, mode)


def write_private_json(
    path: Path,
    data: Mapping[str, Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    payload = dict(data)
    synth_ai_py.write_private_json(str(path), payload)
