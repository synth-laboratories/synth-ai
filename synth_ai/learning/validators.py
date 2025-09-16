from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict
from urllib.parse import urlparse


def validate_training_jsonl(path: str | Path, *, sample_lines: int = 50) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    lines = p.read_text().splitlines()
    if not lines:
        raise ValueError("empty JSONL")
    for i, line in enumerate(lines[: max(1, sample_lines) ], start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"invalid json on line {i}: {e}") from e
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"line {i}: missing messages[] with at least 2 turns")
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if not roles or not isinstance(roles[0], str):
            raise ValueError(f"line {i}: missing first role")
        for m in msgs:
            if not isinstance(m, dict):
                raise ValueError(f"line {i}: non-dict message")
            if not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str) or not m["content"].strip():
                raise ValueError(f"line {i}: invalid role/content")


def validate_task_app_url(url: str, *, name: str = "TASK_APP_BASE_URL") -> None:
    from synth_ai.task.validators import validate_task_app_url as _vt

    _vt(url, name=name)


def validate_trainer_cfg_rl(trainer: Dict[str, Any]) -> None:
    bs = int(trainer.get("batch_size", 1))
    gs = int(trainer.get("group_size", 2))
    if bs < 1:
        raise ValueError("trainer.batch_size must be >= 1")
    if gs < 2:
        raise ValueError("trainer.group_size must be >= 2")
