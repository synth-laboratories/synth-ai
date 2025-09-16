from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FTJobConfig:
    model: str
    training_file_id: str
    n_epochs: int = 1
    batch_size: int = 1
    upload_to_wasabi: bool = True

    def hyperparameters(self) -> Dict[str, Any]:
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return {"n_epochs": int(self.n_epochs), "batch_size": int(self.batch_size)}

    def metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"upload_to_wasabi": bool(self.upload_to_wasabi)}


@dataclass
class RLJobConfig:
    model: str
    task_app_url: str
    trainer_id: str
    batch_size: int = 1
    group_size: int = 2
    job_config_id: Optional[str] = None
    inline_config: Optional[Dict[str, Any]] = None

    def trainer_dict(self) -> Dict[str, Any]:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        return {"batch_size": int(self.batch_size), "group_size": int(self.group_size)}


