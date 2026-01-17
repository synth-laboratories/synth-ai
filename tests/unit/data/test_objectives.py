from __future__ import annotations

from dataclasses import asdict

import pytest
from synth_ai.data.enums import ObjectiveDirection, ObjectiveKey
from synth_ai.data.objectives import (
    EventObjectiveAssignment,
    InstanceObjectiveAssignment,
    OutcomeObjectiveAssignment,
)
from synth_ai.data.objectives_compat import (
    extract_instance_rewards,
    extract_outcome_reward,
)


def test_objective_key_values() -> None:
    assert ObjectiveKey.REWARD.value == "reward"
    assert ObjectiveKey.LATENCY_MS.value == "latency_ms"
    assert ObjectiveKey.COST_USD.value == "cost_usd"


def test_objective_direction_values() -> None:
    assert ObjectiveDirection.MAXIMIZE.value == "maximize"
    assert ObjectiveDirection.MINIMIZE.value == "minimize"


def test_outcome_objective_assignment_serialization() -> None:
    assignment = OutcomeObjectiveAssignment(
        objectives={"reward": 0.75, "latency_ms": 120.0},
        session_id="sess-1",
        trace_id="trace-1",
        metadata={"source": "test"},
    )
    data = asdict(assignment)
    assert data["objectives"]["reward"] == 0.75
    assert data["session_id"] == "sess-1"
    assert data["trace_id"] == "trace-1"
    assert data["metadata"]["source"] == "test"


def test_event_objective_assignment_serialization() -> None:
    assignment = EventObjectiveAssignment(
        event_id="evt-1",
        objectives={"reward": 1.0},
        metadata={"note": "ok"},
    )
    data = asdict(assignment)
    assert data["event_id"] == "evt-1"
    assert data["objectives"]["reward"] == 1.0
    assert data["metadata"]["note"] == "ok"


@pytest.mark.parametrize(
    "payload, expected",
    [
        # outcome_objectives['reward'] is preferred
        ({"outcome_objectives": {"reward": 0.95}}, 0.95),
        # outcome_objectives takes precedence over outcome_reward
        ({"outcome_objectives": {"reward": 0.9}, "outcome_reward": 0.1}, 0.9),
        # outcome_reward is the fallback
        ({"outcome_reward": 0.8}, 0.8),
        # Unknown fields return None
        ({"foo": "bar"}, None),
    ],
)
def test_extract_outcome_reward(payload: dict[str, object], expected: float | None) -> None:
    assert extract_outcome_reward(payload) == expected


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {
                "instance_objectives": [
                    {"objectives": {"reward": 0.1}},
                    {"objectives": {"reward": 0.2}},
                ]
            },
            [0.1, 0.2],
        ),
        (
            {"instance_objectives": [InstanceObjectiveAssignment(1, {"reward": 0.4})]},
            [0.4],
        ),
        ({"instance_rewards": [0.2, 0.3]}, [0.2, 0.3]),
        ({"instance_rewards": ["bad"]}, None),
    ],
)
def test_extract_instance_rewards(
    payload: dict[str, object],
    expected: list[float] | None,
) -> None:
    assert extract_instance_rewards(payload) == expected
