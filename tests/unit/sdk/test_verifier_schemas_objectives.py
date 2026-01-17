from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from synth_ai.sdk.graphs.verifier_schemas import CalibrationExampleInput


def _build_trace(event_count: int) -> dict[str, Any]:
    return {"event_history": [{"type": "step"} for _ in range(event_count)]}


def test_calibration_input_accepts_objectives() -> None:
    trace = _build_trace(2)
    payload = CalibrationExampleInput(
        session_trace=trace,
        event_objectives=[{"reward": 0.2}, {"reward": 0.8}],
        outcome_objectives={"reward": 0.6},
    )
    example = payload.to_dataclass()
    assert example.event_rewards == [0.2, 0.8]
    assert example.outcome_reward == 0.6


def test_calibration_input_objective_length_mismatch() -> None:
    trace = _build_trace(2)
    with pytest.raises(ValidationError, match="event rewards length"):
        CalibrationExampleInput(
            session_trace=trace,
            event_objectives=[{"reward": 0.2}],
            outcome_objectives={"reward": 0.6},
        )
