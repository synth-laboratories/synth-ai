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

    @classmethod
    def from_dict(cls, data: Any) -> ScopeKey:
        if not isinstance(data, dict):
            raise ValueError("ScopeKey must be an object")
        kind_raw = data.get("kind")
        if not isinstance(kind_raw, str):
            raise ValueError("ScopeKey.kind must be a string")
        try:
            kind = ScopeKind(kind_raw)
        except ValueError:
            kind = ScopeKind.CUSTOM
        ident = data.get("id")
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError("ScopeKey.id must be a non-empty string")
        return cls(kind=kind, id=ident)

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


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


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

    @classmethod
    def from_dict(cls, data: Any) -> Lever:
        if not isinstance(data, dict):
            raise ValueError("Lever must be an object")
        kind_raw = data.get("kind", "custom")
        try:
            kind = LeverKind(str(kind_raw))
        except ValueError:
            kind = LeverKind.CUSTOM

        value_format_raw = data.get("value_format", "json")
        try:
            value_format = LeverFormat(str(value_format_raw))
        except ValueError:
            value_format = LeverFormat.CUSTOM

        mutability_raw = data.get("mutability", "optimizer")
        try:
            mutability = LeverMutability(str(mutability_raw))
        except ValueError:
            mutability = LeverMutability.OPTIMIZER
        scope_raw = data.get("scope") or []
        scope = [ScopeKey.from_dict(item) for item in scope_raw] if isinstance(scope_raw, list) else []

        constraints_raw = data.get("constraints")
        constraints = None
        if isinstance(constraints_raw, dict):
            constraints = LeverConstraints(
                allowed_values=constraints_raw.get("allowed_values"),
                min_value=constraints_raw.get("min_value"),
                max_value=constraints_raw.get("max_value"),
                regex=constraints_raw.get("regex"),
                max_bytes=constraints_raw.get("max_bytes"),
                max_tokens=constraints_raw.get("max_tokens"),
                mime_type=constraints_raw.get("mime_type"),
            )

        provenance_raw = data.get("provenance")
        provenance = None
        if isinstance(provenance_raw, dict):
            actor_raw = provenance_raw.get("actor")
            reason_raw = provenance_raw.get("reason")
            if isinstance(actor_raw, str) and isinstance(reason_raw, str):
                try:
                    actor = LeverActor(actor_raw)
                except ValueError:
                    actor = LeverActor.SYSTEM
                provenance = LeverProvenance(
                    actor=actor,
                    reason=reason_raw,
                    source_event_id=provenance_raw.get("source_event_id"),
                    source_trace_id=provenance_raw.get("source_trace_id"),
                )

        version = 1
        try:
            if data.get("version") is not None:
                version = int(data["version"])
        except (TypeError, ValueError):
            version = 1

        created_at = _parse_datetime(data.get("created_at"))
        parent_version = data.get("parent_version")
        try:
            parent_version = int(parent_version) if parent_version is not None else None
        except (TypeError, ValueError):
            parent_version = None

        return cls(
            lever_id=str(data.get("lever_id") or ""),
            kind=kind,
            scope=scope,
            value=data.get("value"),
            value_format=value_format,
            constraints=constraints,
            mutability=mutability,
            provenance=provenance,
            version=version,
            created_at=created_at,
            parent_version=parent_version,
        )

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


@dataclass(slots=True)
class MiproLeverSummary:
    prompt_lever_id: Optional[str] = None
    candidate_lever_versions: dict[str, int] = field(default_factory=dict)
    best_candidate_id: Optional[str] = None
    selected_candidate_id: Optional[str] = None
    baseline_candidate_id: Optional[str] = None
    lever_count: Optional[int] = None
    mutation_count: Optional[int] = None
    latest_version: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Any) -> Optional[MiproLeverSummary]:
        if not isinstance(data, dict):
            return None
        prompt_lever_id = data.get("prompt_lever_id")
        if not isinstance(prompt_lever_id, str):
            prompt_lever_id = None
        candidate_versions_raw = data.get("candidate_lever_versions")
        candidate_versions: dict[str, int] = {}
        if isinstance(candidate_versions_raw, dict):
            for k, v in candidate_versions_raw.items():
                try:
                    candidate_versions[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue

        def _opt_str(key: str) -> Optional[str]:
            value = data.get(key)
            return value if isinstance(value, str) and value.strip() else None

        def _opt_int(key: str) -> Optional[int]:
            value = data.get(key)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            prompt_lever_id=prompt_lever_id,
            candidate_lever_versions=candidate_versions,
            best_candidate_id=_opt_str("best_candidate_id"),
            selected_candidate_id=_opt_str("selected_candidate_id"),
            baseline_candidate_id=_opt_str("baseline_candidate_id"),
            lever_count=_opt_int("lever_count"),
            mutation_count=_opt_int("mutation_count"),
            latest_version=_opt_int("latest_version"),
        )
