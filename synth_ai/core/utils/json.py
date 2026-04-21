from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strip_json_comments(raw: str) -> str:
    """Remove // and /* */ comments from JSONC text."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(raw):
        char = raw[i]
        next_char = raw[i + 1] if i + 1 < len(raw) else ""
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and next_char == "/":
            i += 2
            while i < len(raw) and raw[i] not in "\r\n":
                i += 1
            continue
        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(raw) and not (raw[i] == "*" and raw[i + 1] == "/"):
                i += 1
            i = min(i + 2, len(raw))
            continue
        out.append(char)
        i += 1
    return "".join(out)


def create_and_write_json(path: Path, content: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json_to_dict(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload
