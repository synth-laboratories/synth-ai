"""Payload helpers for Graph Evolve requests."""

from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_snapshot_id(
    *,
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Optional[str]:
    if prompt_snapshot_id and graph_snapshot_id:
        raise ValueError("Provide only one of prompt_snapshot_id or graph_snapshot_id.")
    return graph_snapshot_id or prompt_snapshot_id


def build_graph_record_payload(
    *,
    job_id: str,
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"job_id": job_id}
    snapshot_id = resolve_snapshot_id(
        prompt_snapshot_id=prompt_snapshot_id,
        graph_snapshot_id=graph_snapshot_id,
    )
    if snapshot_id:
        payload["prompt_snapshot_id"] = snapshot_id
    return payload


def build_inference_payload(
    *,
    job_id: str,
    input_data: Dict[str, Any],
    model: Optional[str],
    prompt_snapshot_id: Optional[str],
    graph_snapshot_id: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"job_id": job_id, "input": input_data}
    if model:
        payload["model"] = model
    snapshot_id = resolve_snapshot_id(
        prompt_snapshot_id=prompt_snapshot_id,
        graph_snapshot_id=graph_snapshot_id,
    )
    if snapshot_id:
        payload["prompt_snapshot_id"] = snapshot_id
    return payload


__all__ = [
    "resolve_snapshot_id",
    "build_graph_record_payload",
    "build_inference_payload",
]
