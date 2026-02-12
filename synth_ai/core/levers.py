from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ScopeKind(str, Enum):
    ORG = "org"
    SYSTEM = "system"
    RUN = "run"
    BRANCH = "branch"
    ITERATION = "iteration"
    CANDIDATE = "candidate"
    STAGE = "stage"
    SEED = "seed"
    ROLLOUT = "rollout"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


@dataclass(slots=True)
class ScopeKey:
    kind: ScopeKind
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "id": self.id}


class LeverKind(str, Enum):
    PROMPT = "prompt"
    CONTEXT = "context"
    CODE = "code"
    CONSTRAINT = "constraint"
    NOTE = "note"
    SPEC = "spec"
    GRAPH_YAML = "graph_yaml"
    VARIABLE = "variable"
    EXPERIMENT = "experiment"
    CUSTOM = "custom"


class LeverFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    FILE = "file"
    CUSTOM = "custom"


class LeverMutability(str, Enum):
    OPTIMIZER = "optimizer"
    HUMAN = "human"
    SYSTEM = "system"


class LeverActor(str, Enum):
    OPTIMIZER = "optimizer"
    HUMAN = "human"
    SYSTEM = "system"


@dataclass(slots=True)
class LeverConstraints:
    allowed_values: Optional[list[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    regex: Optional[str] = None
    max_bytes: Optional[int] = None
    max_tokens: Optional[int] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.allowed_values is not None:
            payload["allowed_values"] = self.allowed_values
        if self.min_value is not None:
            payload["min_value"] = self.min_value
        if self.max_value is not None:
            payload["max_value"] = self.max_value
        if self.regex is not None:
            payload["regex"] = self.regex
        if self.max_bytes is not None:
            payload["max_bytes"] = self.max_bytes
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        return payload


@dataclass(slots=True)
class LeverProvenance:
    actor: LeverActor
    reason: str
    source_event_id: Optional[str] = None
    source_trace_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"actor": self.actor.value, "reason": self.reason}
        if self.source_event_id is not None:
            payload["source_event_id"] = self.source_event_id
        if self.source_trace_id is not None:
            payload["source_trace_id"] = self.source_trace_id
        return payload


@dataclass(slots=True)
class Lever:
    lever_id: str
    kind: LeverKind
    scope: list[ScopeKey]
    value: Any
    value_format: LeverFormat
    constraints: Optional[LeverConstraints] = None
    mutability: LeverMutability = LeverMutability.OPTIMIZER
    provenance: Optional[LeverProvenance] = None
    version: int = 1
    created_at: Optional[datetime] = None
    parent_version: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lever_id": self.lever_id,
            "kind": self.kind.value,
            "scope": [item.to_dict() for item in self.scope],
            "value": self.value,
            "value_format": self.value_format.value,
            "mutability": self.mutability.value,
            "version": self.version,
        }
        if self.constraints is not None:
            payload["constraints"] = self.constraints.to_dict()
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        if self.parent_version is not None:
            payload["parent_version"] = self.parent_version
        return payload


@dataclass(slots=True)
class LeverSnapshot:
    lever_id: str
    resolved_scope: list[ScopeKey]
    version: int
    value: Any
    applied_at: Optional[datetime] = None


@dataclass(slots=True)
class LeverMutation:
    lever_id: str
    parent_version: int
    new_version: int
    mutation_type: str
    delta: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    optimizer_id: str = ""
