"""Admitted resident capacity. See SDK swarms_scientific_delivery.md."""
from typing import Literal
from pydantic import Field, StrictInt
from synth_ai.types.base import ResponseBase


class EffectiveLimits(ResponseBase):
    actors: StrictInt = Field(ge=0)
    branches: StrictInt = Field(ge=0)
    peers: StrictInt = Field(ge=0)
    shared_sessions: StrictInt = Field(ge=0, le=32)
    placement: Literal["isolated", "shared", "none", "mixed", "unresolved"]
