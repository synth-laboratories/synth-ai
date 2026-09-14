"""Runs carry the Intern runtime that launched them."""

from __future__ import annotations

from synth_ai.sdk.research.contracts.run_state import ManagedResearchRun


def test_run_parses_launch_origin() -> None:
    run = ManagedResearchRun.from_wire(
        {
            "run_id": "run-1",
            "project_id": "proj-1",
            "public_state": "running",
            "origin_runtime_kind": "sync",
            "origin_runtime_id": "sync-session-1",
        }
    )

    assert run.origin_runtime_kind == "sync"
    assert run.origin_runtime_id == "sync-session-1"


def test_run_without_intern_origin_reads_none() -> None:
    run = ManagedResearchRun.from_wire(
        {"run_id": "run-1", "project_id": "proj-1", "public_state": "running"}
    )

    assert run.origin_runtime_kind is None
    assert run.origin_runtime_id is None
