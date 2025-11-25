from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_ai.sdk.learning.sft import SFTDataError, parse_jsonl_line


def validate_training_jsonl(path: str | Path, *, sample_lines: int = 50) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    max_samples = max(1, sample_lines)
    non_empty_lines = 0

    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            non_empty_lines += 1
            if non_empty_lines > max_samples:
                break
            try:
                parse_jsonl_line(stripped, min_messages=2)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json on line {lineno}: {exc}") from exc
            except SFTDataError as exc:
                raise ValueError(f"line {lineno}: {exc}") from exc

    if non_empty_lines == 0:
        raise ValueError("empty JSONL")


def validate_task_app_url(url: str, *, name: str = "TASK_APP_BASE_URL") -> None:
    from synth_ai.sdk.task.validators import validate_task_app_url as _vt

    try:
        _vt(url)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def validate_trainer_cfg_rl(trainer: dict[str, Any]) -> None:
    bs = int(trainer.get("batch_size", 1))
    gs = int(trainer.get("group_size", 2))
    if bs < 1:
        raise ValueError("trainer.batch_size must be >= 1")
    if gs < 2:
        raise ValueError("trainer.group_size must be >= 2")
