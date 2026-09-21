"""Fresh resource-settlement read for one run.

Mirrors backend ``RunResourceSettlementResponse``
(``services/smr/ownership/contracts.py``). The backend forbids extra fields, so
an unknown key here is contract drift and fails loudly rather than being
ignored. ``settled`` only speaks for resources the run registered; whether the
registered inventory is complete is ``coverage_complete``. A run with
``coverage="untracked"`` makes no settlement claim at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SettlementCoverage(StrEnum):
    UNTRACKED = "untracked"
    EXPLICIT_V1 = "explicit-v1"


class SettlementScope(StrEnum):
    ROOT_TREE = "root_tree"
    OWNED_SUBTREE = "owned_subtree"


_FIELDS = frozenset(
    {
        "run_id",
        "observed_at",
        "coverage",
        "settled",
        "registered_tree_settled",
        "coverage_complete",
        "scope_kind",
        "root_run_id",
        "edge_id",
        "pending",
        "unknown",
        "confirmed",
        "excluded",
        "root_confirmed",
    }
)


def _bool(mapping: Mapping[str, object], key: str, *, default: bool | None = None) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"resource settlement {key} must be a boolean")
    return value


def _optional_bool(mapping: Mapping[str, object], key: str) -> bool | None:
    value = mapping.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"resource settlement {key} must be a boolean or null")


def _optional_count(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"resource settlement {key} must be a non-negative integer or null")
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"resource settlement {key} must be a string or null")


@dataclass(frozen=True)
class RunResourceSettlement:
    run_id: str
    observed_at: datetime
    coverage: SettlementCoverage
    settled: bool
    registered_tree_settled: bool = False
    coverage_complete: bool = False
    scope_kind: SettlementScope | None = None
    root_run_id: str | None = None
    edge_id: str | None = None
    pending: int | None = None
    unknown: int | None = None
    confirmed: int | None = None
    excluded: int | None = None
    root_confirmed: bool | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunResourceSettlement:
        if not isinstance(payload, Mapping):
            raise ValueError("resource settlement must be an object")
        unknown_keys = set(payload) - _FIELDS
        if unknown_keys:
            raise ValueError(f"resource settlement has unknown fields: {sorted(unknown_keys)}")
        run_id = payload.get("run_id")
        observed_at = payload.get("observed_at")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("resource settlement run_id is required")
        if not isinstance(observed_at, str):
            raise ValueError("resource settlement observed_at is required")
        scope_kind = _optional_str(payload, "scope_kind")
        return cls(
            run_id=run_id,
            observed_at=datetime.fromisoformat(observed_at),
            coverage=SettlementCoverage(payload.get("coverage")),
            settled=_bool(payload, "settled"),
            registered_tree_settled=_bool(payload, "registered_tree_settled", default=False),
            coverage_complete=_bool(payload, "coverage_complete", default=False),
            scope_kind=SettlementScope(scope_kind) if scope_kind is not None else None,
            root_run_id=_optional_str(payload, "root_run_id"),
            edge_id=_optional_str(payload, "edge_id"),
            pending=_optional_count(payload, "pending"),
            unknown=_optional_count(payload, "unknown"),
            confirmed=_optional_count(payload, "confirmed"),
            excluded=_optional_count(payload, "excluded"),
            root_confirmed=_optional_bool(payload, "root_confirmed"),
        )


__all__ = ["RunResourceSettlement", "SettlementCoverage", "SettlementScope"]
