from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from synth_ai.core.levers import ScopeKey, _parse_datetime


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

    @classmethod
    def from_dict(cls, data: Any) -> Sensor:
        if not isinstance(data, dict):
            raise ValueError("Sensor must be an object")
        kind_raw = data.get("kind", "custom")
        try:
            kind = SensorKind(str(kind_raw))
        except ValueError:
            kind = SensorKind.CUSTOM
        scope_raw = data.get("scope") or []
        scope = [ScopeKey.from_dict(item) for item in scope_raw] if isinstance(scope_raw, list) else []
        timestamp = _parse_datetime(data.get("timestamp"))
        trace_id = data.get("trace_id") if isinstance(data.get("trace_id"), str) else None
        units = data.get("units") if isinstance(data.get("units"), str) else None
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return cls(
            sensor_id=str(data.get("sensor_id") or ""),
            kind=kind,
            scope=scope,
            value=data.get("value"),
            units=units,
            timestamp=timestamp,
            trace_id=trace_id,
            metadata=metadata,
        )

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

    @classmethod
    def from_dict(cls, data: Any) -> SensorFrame:
        if not isinstance(data, dict):
            raise ValueError("SensorFrame must be an object")
        scope_raw = data.get("scope") or []
        scope = [ScopeKey.from_dict(item) for item in scope_raw] if isinstance(scope_raw, list) else []
        sensors_raw = data.get("sensors") or []
        sensors = [Sensor.from_dict(item) for item in sensors_raw] if isinstance(sensors_raw, list) else []
        lever_versions_raw = data.get("lever_versions")
        lever_versions: dict[str, int] = {}
        if isinstance(lever_versions_raw, dict):
            for k, v in lever_versions_raw.items():
                try:
                    lever_versions[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
        trace_ids_raw = data.get("trace_ids") or []
        trace_ids = [str(x) for x in trace_ids_raw if isinstance(x, (str, int))] if isinstance(trace_ids_raw, list) else []
        return cls(
            scope=scope,
            sensors=sensors,
            lever_versions=lever_versions,
            trace_ids=trace_ids,
            created_at=_parse_datetime(data.get("created_at")),
            frame_id=data.get("frame_id") if isinstance(data.get("frame_id"), str) else None,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

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


@dataclass(slots=True)
class SensorFrameSummary:
    frame_id: str
    created_at: Optional[datetime] = None
    sensor_count: Optional[int] = None
    sensor_kinds: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    lever_versions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> Optional[SensorFrameSummary]:
        if not isinstance(data, dict):
            return None
        frame_id = data.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            return None
        created_at = _parse_datetime(data.get("created_at"))
        sensor_count = data.get("sensor_count")
        try:
            sensor_count = int(sensor_count) if sensor_count is not None else None
        except (TypeError, ValueError):
            sensor_count = None

        sensor_kinds_raw = data.get("sensor_kinds") or []
        sensor_kinds = [str(x) for x in sensor_kinds_raw if isinstance(x, (str, int))] if isinstance(sensor_kinds_raw, list) else []
        trace_ids_raw = data.get("trace_ids") or []
        trace_ids = [str(x) for x in trace_ids_raw if isinstance(x, (str, int))] if isinstance(trace_ids_raw, list) else []
        lever_versions_raw = data.get("lever_versions")
        lever_versions: dict[str, int] = {}
        if isinstance(lever_versions_raw, dict):
            for k, v in lever_versions_raw.items():
                try:
                    lever_versions[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            frame_id=frame_id,
            created_at=created_at,
            sensor_count=sensor_count,
            sensor_kinds=sensor_kinds,
            trace_ids=trace_ids,
            lever_versions=lever_versions,
        )
