"""Usage receipt for one Intern Sync session or Async assignment.

Mirrors backend ``InternSessionUsageResponse``
(``app/api/v1/managed_research/schemas/intern_usage.py``). ``spend_cents`` is
settled run spend only; while ``billing.billing_state`` is not ``settled`` the
total is incomplete, never a final $0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class InternBillingState(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    STALLED = "stalled"
    FAILED = "failed"
    OBSERVABILITY_ONLY = "observability_only"
    NOT_APPLICABLE = "not_applicable"


def _int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"intern usage {key} must be an integer")
    return value


@dataclass(frozen=True)
class InternUsageBilling:
    billing_state: InternBillingState
    stalled_row_count: int
    stalled_spend_cents: int
    billing_failed_row_count: int
    show_stalled_banner: bool
    message: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> InternUsageBilling:
        if not isinstance(payload, Mapping):
            raise ValueError("intern usage billing must be an object")
        banner = payload.get("show_stalled_banner")
        if not isinstance(banner, bool):
            raise ValueError("intern usage billing show_stalled_banner must be a boolean")
        message = payload.get("message")
        if message is not None and not isinstance(message, str):
            raise ValueError("intern usage billing message must be a string or null")
        return cls(
            billing_state=InternBillingState(payload.get("billing_state")),
            stalled_row_count=_int(payload, "stalled_row_count"),
            stalled_spend_cents=_int(payload, "stalled_spend_cents"),
            billing_failed_row_count=_int(payload, "billing_failed_row_count"),
            show_stalled_banner=banner,
            message=message,
        )


@dataclass(frozen=True)
class InternSessionUsage:
    origin_runtime_kind: str
    origin_runtime_id: str
    org_id: str
    spend_cents: int
    token_count: int
    run_count: int
    billing: InternUsageBilling
    bindings: tuple[dict[str, object], ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def run_ids(self) -> tuple[str, ...]:
        """Runs this Intern runtime launched, as recorded in the receipt."""
        return tuple(str(run_id) for run_id in self.metadata.get("run_ids", ()) or ())

    @classmethod
    def from_wire(cls, payload: object) -> InternSessionUsage:
        if not isinstance(payload, Mapping):
            raise ValueError("intern usage must be an object")
        kind = payload.get("origin_runtime_kind")
        if kind not in {"sync", "async"}:
            raise ValueError(f"intern usage origin_runtime_kind is invalid: {kind!r}")
        runtime_id = payload.get("origin_runtime_id")
        org_id = payload.get("org_id")
        if not isinstance(runtime_id, str) or not isinstance(org_id, str):
            raise ValueError("intern usage origin_runtime_id and org_id are required")
        bindings = payload.get("bindings") or []
        metadata = payload.get("metadata") or {}
        if not isinstance(bindings, list) or not isinstance(metadata, Mapping):
            raise ValueError("intern usage bindings must be a list and metadata an object")
        return cls(
            origin_runtime_kind=kind,
            origin_runtime_id=runtime_id,
            org_id=org_id,
            spend_cents=_int(payload, "spend_cents"),
            token_count=_int(payload, "token_count"),
            run_count=_int(payload, "run_count"),
            billing=InternUsageBilling.from_wire(payload.get("billing")),
            bindings=tuple(dict(item) for item in bindings),
            metadata=dict(metadata),
        )


__all__ = ["InternBillingState", "InternSessionUsage", "InternUsageBilling"]
