from __future__ import annotations

from synth_ai.sdk.research.contracts.types import KickoffContract


def test_kickoff_contract_preserves_trace_capture_round_trip() -> None:
    trace_capture = {
        "schema_version": "smr.trace-capture.v1",
        "mode": "required",
        "producer": "evals.scripts.run_bench_matrix",
        "benchmark": "factorybench",
    }

    contract = KickoffContract.from_wire(
        {
            "schema_version": 1,
            "contract_kind": "staged_smr_kickoff_contract",
            "run_objective": "Trace the recurring Factory",
            "trace_capture": trace_capture,
        }
    )

    assert contract.trace_capture == trace_capture
    assert contract.to_wire()["trace_capture"] == trace_capture


def test_kickoff_contract_omits_absent_trace_capture() -> None:
    contract = KickoffContract.from_wire(
        {
            "schema_version": 1,
            "contract_kind": "staged_smr_kickoff_contract",
            "run_objective": "No trace capture requested",
        }
    )

    assert contract.trace_capture is None
    assert "trace_capture" not in contract.to_wire()
