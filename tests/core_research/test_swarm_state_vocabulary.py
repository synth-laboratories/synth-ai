"""The client must accept every state the deployed backend actually sends.

`swarms.activity` raised `ValueError: 'completed' is not a valid SwarmState`
against a real staging run: the backend reports `completed`, the enum did not
have it, and the strict decode turned a healthy read into a client exception.
An unknown state is a contract question, but a state the backend has been
sending all along is just a missing member.
"""

from __future__ import annotations

import pytest
from synth_ai.core.research.contracts.swarms import SwarmState

# Every value the Managed Research API can report for a run's public state.
BACKEND_RUN_STATES = (
    "queued",
    "active",
    "paused",
    "finalizing",
    "done",
    "completed",
    "partial",
    "failed",
    "stopped",
    "canceled",
)

TERMINAL_BACKEND_STATES = frozenset(
    {"done", "completed", "partial", "failed", "stopped", "canceled"}
)


@pytest.mark.parametrize("value", BACKEND_RUN_STATES)
def test_state_decodes(value: str) -> None:
    assert SwarmState(value).value == value


@pytest.mark.parametrize("value", sorted(TERMINAL_BACKEND_STATES))
def test_terminal_states_are_terminal(value: str) -> None:
    assert SwarmState(value).is_terminal is True


@pytest.mark.parametrize(
    "value", sorted(set(BACKEND_RUN_STATES) - TERMINAL_BACKEND_STATES)
)
def test_live_states_are_not_terminal(value: str) -> None:
    assert SwarmState(value).is_terminal is False
