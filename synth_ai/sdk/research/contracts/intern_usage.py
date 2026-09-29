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
    """Backend billing settlement states for Intern runtime usage."""
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
    """Billing settlement state and stalled/failed-row diagnostics.

    ```python
    from synth_ai.sdk.research.contracts.intern_usage import InternUsageBilling

    billing = InternUsageBilling.from_wire({
        "billing_state": "pending", "stalled_row_count": 0,
        "stalled_spend_cents": 0, "billing_failed_row_count": 0,
        "show_stalled_banner": False,
    })
    assert billing.billing_state == "pending"
    ```
    """
    #: Settlement state of the billing receipt; only settled is a final total.
    billing_state: InternBillingState
    #: Number of billing rows reported as stalled.
    stalled_row_count: int
    #: Spend represented by stalled rows, in cents.
    stalled_spend_cents: int
    #: Number of billing rows reported as failed.
    billing_failed_row_count: int
    #: Backend flag indicating a stalled-billing notice should be shown.
    show_stalled_banner: bool
    #: Backend billing explanation, when provided.
    message: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> InternUsageBilling:
        """Parse the backend wire representation of InternUsageBilling.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed InternUsageBilling with backend values preserved.

        Raises:
            ValueError: Billing is not an object, its state is unknown, or its flags/counters have invalid types.
        """
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
    """Usage receipt for an Intern runtime; unsettled billing makes spend incomplete.

    ```python
    from synth_ai.sdk.research.contracts.intern_usage import InternSessionUsage

    usage = InternSessionUsage.from_wire({
        "origin_runtime_kind": "sync", "origin_runtime_id": "session-1",
        "org_id": "org-1", "spend_cents": 0, "token_count": 0, "run_count": 0,
        "billing": {"billing_state": "pending", "stalled_row_count": 0,
            "stalled_spend_cents": 0, "billing_failed_row_count": 0,
            "show_stalled_banner": False},
    })
    # Pending billing means this zero spend is not a final total.
    assert usage.billing.billing_state == "pending"
    ```
    """
    #: Originating Intern runtime kind: sync or async.
    origin_runtime_kind: str
    #: Originating Sync session or Async assignment ID.
    origin_runtime_id: str
    #: Organization that owns the runtime usage.
    org_id: str
    #: Settled run spend in cents; incomplete while billing is not settled.
    spend_cents: int
    #: Token count reported for the runtime.
    token_count: int
    #: Number of runs reported for the runtime.
    run_count: int
    #: Billing state and stalled/failed-row diagnostics.
    billing: InternUsageBilling
    #: Backend resource-binding records included in the receipt.
    bindings: tuple[dict[str, object], ...] = field(default_factory=tuple)
    #: Additional receipt metadata, including recorded run_ids when present.
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def run_ids(self) -> tuple[str, ...]:
        """Runs this Intern runtime launched, as recorded in the receipt.

        Returns:
            Run IDs recorded in receipt metadata, or an empty tuple when absent.
        """
        return tuple(str(run_id) for run_id in self.metadata.get("run_ids", ()) or ())

    @classmethod
    def from_wire(cls, payload: object) -> InternSessionUsage:
        """Parse the backend wire representation of InternSessionUsage.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed InternSessionUsage with backend values preserved.

        Raises:
            ValueError: The runtime kind, required identifiers, counts, billing object, bindings, or metadata are malformed.
        """
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
