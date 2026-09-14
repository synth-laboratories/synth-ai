"""Typed reads for run resource settlement and Intern session usage."""

from __future__ import annotations

import pytest

from synth_ai.sdk.research.contracts.intern_usage import InternBillingState, InternSessionUsage
from synth_ai.sdk.research.contracts.resource_settlement import (
    RunResourceSettlement,
    SettlementCoverage,
    SettlementScope,
)

_SETTLEMENT = {
    "run_id": "run-1",
    "observed_at": "2026-09-11T20:00:00+00:00",
    "coverage": "explicit-v1",
    "settled": True,
    "registered_tree_settled": True,
    "coverage_complete": False,
    "scope_kind": "root_tree",
    "root_run_id": "run-1",
    "edge_id": None,
    "pending": 0,
    "unknown": 1,
    "confirmed": 3,
    "excluded": 0,
    "root_confirmed": True,
}


def test_settlement_parses_every_backend_field() -> None:
    settlement = RunResourceSettlement.from_wire(_SETTLEMENT)

    assert settlement.coverage is SettlementCoverage.EXPLICIT_V1
    assert settlement.scope_kind is SettlementScope.ROOT_TREE
    assert settlement.settled and settlement.registered_tree_settled
    assert settlement.coverage_complete is False
    assert (settlement.pending, settlement.unknown, settlement.confirmed) == (0, 1, 3)


def test_settlement_rejects_contract_drift() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        RunResourceSettlement.from_wire({**_SETTLEMENT, "new_field": 1})
    with pytest.raises(ValueError):
        RunResourceSettlement.from_wire({**_SETTLEMENT, "coverage": "complete"})
    with pytest.raises(ValueError):
        RunResourceSettlement.from_wire({**_SETTLEMENT, "settled": "yes"})


def test_untracked_run_makes_no_settlement_claim() -> None:
    settlement = RunResourceSettlement.from_wire(
        {
            "run_id": "run-2",
            "observed_at": "2026-09-11T20:00:00+00:00",
            "coverage": "untracked",
            "settled": False,
        }
    )

    assert settlement.coverage is SettlementCoverage.UNTRACKED
    assert settlement.pending is None and settlement.scope_kind is None


def test_intern_usage_parses_receipt_and_launched_runs() -> None:
    usage = InternSessionUsage.from_wire(
        {
            "origin_runtime_kind": "sync",
            "origin_runtime_id": "sync-1",
            "org_id": "org-1",
            "spend_cents": 42,
            "token_count": 0,
            "run_count": 2,
            "billing": {
                "billing_state": "pending",
                "stalled_row_count": 0,
                "stalled_spend_cents": 0,
                "billing_failed_row_count": 0,
                "message": None,
                "show_stalled_banner": False,
            },
            "bindings": [],
            "metadata": {"run_ids": ["run-a", "run-b"], "direct_inference": {}},
        }
    )

    assert usage.billing.billing_state is InternBillingState.PENDING
    assert usage.run_ids == ("run-a", "run-b")
    assert usage.spend_cents == 42


def test_intern_usage_rejects_unknown_origin_kind() -> None:
    with pytest.raises(ValueError, match="origin_runtime_kind"):
        InternSessionUsage.from_wire({"origin_runtime_kind": "batch"})
