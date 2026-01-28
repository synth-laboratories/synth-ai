"""Rust-backed normalization helpers for data models."""

from __future__ import annotations

from typing import Any

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.rust helpers.") from exc


def _require() -> Any:
    return synth_ai_py


def normalize_rubric(payload: Any) -> Any:
    return _require().normalize_rubric(payload)


def normalize_judgement(payload: Any) -> Any:
    return _require().normalize_judgement(payload)


def normalize_objective_spec(payload: Any) -> Any:
    return _require().normalize_objective_spec(payload)


def normalize_reward_observation(payload: Any) -> Any:
    return _require().normalize_reward_observation(payload)


def normalize_outcome_reward_record(payload: Any) -> Any:
    return _require().normalize_outcome_reward_record(payload)


def normalize_event_reward_record(payload: Any) -> Any:
    return _require().normalize_event_reward_record(payload)


def normalize_context_override(payload: Any) -> Any:
    return _require().normalize_context_override(payload)


def normalize_artifact(payload: Any) -> Any:
    return _require().normalize_artifact(payload)


def normalize_trace(payload: Any) -> Any:
    return _require().normalize_trace(payload)


def normalize_llm_call_record(payload: Any) -> Any:
    return _require().normalize_llm_call_record(payload)


__all__ = [
    "normalize_rubric",
    "normalize_judgement",
    "normalize_objective_spec",
    "normalize_reward_observation",
    "normalize_outcome_reward_record",
    "normalize_event_reward_record",
    "normalize_context_override",
    "normalize_artifact",
    "normalize_trace",
    "normalize_llm_call_record",
]
