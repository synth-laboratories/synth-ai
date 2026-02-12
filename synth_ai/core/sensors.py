from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from synth_ai.core.levers import ScopeKey


class SensorKind(str, Enum):
    REWARD = "reward"
    TIMING = "timing"
    ROLLOUT = "rollout"
    RESOURCE = "resource"
    SAFETY = "safety"
    QUALITY = "quality"
    TRACE = "trace"
    CONTEXT_APPLY = "context_apply"
    EXPERIMENT = "experiment"
    CUSTOM = "custom"


@dataclass(slots=True)
class Sensor:
    sensor_id: str
    kind: SensorKind
    scope: list[ScopeKey]
    value: Any
    units: Optional[str] = None
    timestamp: Optional[datetime] = None
    trace_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sensor_id": self.sensor_id,
            "kind": self.kind.value,
            "scope": [item.to_dict() for item in self.scope],
            "value": self.value,
            "metadata": self.metadata,
        }
        if self.units is not None:
            payload["units"] = self.units
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp.isoformat()
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        return payload


@dataclass(slots=True)
class SensorFrame:
    scope: list[ScopeKey]
    sensors: list[Sensor]
    lever_versions: dict[str, int] = field(default_factory=dict)
    trace_ids: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    frame_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scope": [item.to_dict() for item in self.scope],
            "sensors": [sensor.to_dict() for sensor in self.sensors],
            "lever_versions": self.lever_versions,
            "trace_ids": self.trace_ids,
            "metadata": self.metadata,
        }
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        if self.frame_id is not None:
            payload["frame_id"] = self.frame_id
        return payload
