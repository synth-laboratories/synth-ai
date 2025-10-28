from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _ensure_positive(value: Any, *, name: str) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if ivalue < 1:
        raise ValueError(f"{name} must be >= 1")
    return ivalue


@dataclass(slots=True)
class RLJobConfig:
    model: str
    task_app_url: str
    trainer_id: str
    batch_size: int = 1
    group_size: int = 2
    job_config_id: str | None = None
    inline_config: dict[str, Any] | None = None

    def trainer_dict(self) -> dict[str, Any]:
        return {
            "batch_size": _ensure_positive(self.batch_size, name="trainer.batch_size"),
            "group_size": _ensure_positive(self.group_size, name="trainer.group_size"),
        }
