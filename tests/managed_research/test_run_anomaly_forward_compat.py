"""Forward-compat parsing of run anomalies: unknown wire kinds must not kill polling."""

from __future__ import annotations

from synth_ai.managed_research.models.run_observability import (
    RunAnomaly,
    RunAnomalyKind,
    RunObservabilitySnapshot,
)


def _snapshot_payload(anomalies: list[dict[str, object]]) -> dict[str, object]:
    """Minimal valid run observability snapshot wire payload."""
    return {
        "schema_version": "run_observability.v1",
        "project_id": "proj_x",
        "run_id": "run_x",
        "generated_at": "2026-07-20T00:00:00Z",
        "run": {"project_id": "proj_x", "run_id": "run_x", "public_state": "running"},
        "lifecycle": {
            "authority_phase": "executing",
            "terminal_phase": "none",
            "dispatch": {"owner": "backend", "pool_id": "pool_x", "host_kind": "hosted"},
        },
        "public_state": "running",
        "run_contract": {
            "schema_version": "run_contract.v1",
            "project_id": "proj_x",
            "run_id": "run_x",
            "public_state": "running",
            "terminal": False,
            "lifecycle": {"phase": "executing"},
            "finalization": {"status": "pending"},
            "recovery": {"status": "idle"},
            "tasks": {"total": 1, "terminal": 0, "nonterminal": 1},
            "artifacts": {"readiness": "pending"},
            "execution_route": {"route": "hosted"},
            "work_products": {},
            "trained_models": {},
            "container_eval_packages": {},
            "incidents": {"unresolved": 0},
            "diagnostics": {},
        },
        "candidate_publication": {"outcome": "running"},
        "actors": {},
        "tasks": {},
        "runtime": {},
        "cursor": {},
        "anomalies": anomalies,
    }


def test_known_kind_parses_to_member() -> None:
    anomaly = RunAnomaly.from_wire(
        {"kind": "mcp_unreachable", "detail": "runtime MCP endpoint refused connection"}
    )
    assert anomaly.kind is RunAnomalyKind.MCP_UNREACHABLE
    assert anomaly.raw_kind == "mcp_unreachable"
    assert anomaly.detail == "runtime MCP endpoint refused connection"
    assert not anomaly.is_unknown_kind


def test_actor_binding_projection_divergence_is_known() -> None:
    anomaly = RunAnomaly.from_wire(
        {"kind": "actor_binding_projection_divergence", "detail": "projection diverged"}
    )
    assert anomaly.kind is RunAnomalyKind.ACTOR_BINDING_PROJECTION_DIVERGENCE
    assert not anomaly.is_unknown_kind


def test_unknown_kind_parses_and_preserves_raw_value_and_detail() -> None:
    anomaly = RunAnomaly.from_wire(
        {
            "kind": "quantum_flux_capacitor_drift",
            "detail": "synthetic future anomaly emitted by a newer backend",
        }
    )
    assert anomaly.kind is RunAnomalyKind.UNKNOWN
    assert anomaly.raw_kind == "quantum_flux_capacitor_drift"
    assert anomaly.detail == "synthetic future anomaly emitted by a newer backend"
    assert anomaly.is_unknown_kind


def test_snapshot_with_new_anomaly_string_parses_end_to_end() -> None:
    snapshot = RunObservabilitySnapshot.from_wire(
        _snapshot_payload(
            [
                {"kind": "actor_binding_projection_divergence", "detail": "known kind"},
                {"kind": "quantum_flux_capacitor_drift", "detail": "future kind"},
            ]
        )
    )
    known, unknown = snapshot.anomalies
    assert known.kind is RunAnomalyKind.ACTOR_BINDING_PROJECTION_DIVERGENCE
    assert not known.is_unknown_kind
    assert unknown.kind is RunAnomalyKind.UNKNOWN
    assert unknown.raw_kind == "quantum_flux_capacitor_drift"
    assert unknown.detail == "future kind"
    assert unknown.is_unknown_kind
