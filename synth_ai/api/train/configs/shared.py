from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtraModel(BaseModel):
    """Base model that tolerates unknown keys so configs keep forward compatibility."""

    model_config = ConfigDict(extra="allow")


class AlgorithmConfig(ExtraModel):
    type: str
    method: str
    variety: str


class ComputeConfig(ExtraModel):
    gpu_type: str
    gpu_count: int
    nodes: int | None = None


__all__ = ["ExtraModel", "AlgorithmConfig", "ComputeConfig"]
