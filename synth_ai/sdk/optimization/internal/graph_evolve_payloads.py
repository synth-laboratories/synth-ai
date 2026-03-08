"""Payload helpers for Graph Evolve requests."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.graph_evolve_payloads.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "build_graph_evolve_inference_payload"):
        raise RuntimeError(
            "Rust core graph evolve payload helpers required; synth_ai_py is unavailable."
        )
    return synth_ai_py


def resolve_snapshot_id(
    *,
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Optional[str]:
    rust = _require_rust()
    return rust.resolve_graph_evolve_snapshot_id(prompt_snapshot_id, graph_snapshot_id)


def build_graph_record_payload(
    *,
    job_id: str,
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.build_graph_evolve_graph_record_payload(
        job_id, prompt_snapshot_id, graph_snapshot_id
    )


def build_inference_payload(
    *,
    job_id: str,
    input_data: Dict[str, Any],
    model: Optional[str],
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.build_graph_evolve_inference_payload(
        job_id, input_data, model, prompt_snapshot_id, graph_snapshot_id
    )


__all__ = [
    "resolve_snapshot_id",
    "build_graph_record_payload",
    "build_inference_payload",
]
