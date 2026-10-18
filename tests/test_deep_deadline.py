"""Bounded DEEP search deadline on KeyedSearchOptions (SYN-4612); consumer-first wire shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from synth_ai.sdk.index.types.search import KeyedSearchOptions
from synth_ai.types.money import Money

CHARGE = Money(coefficient=1000, scale=2, currency="USD")


def _options(**fields: object) -> KeyedSearchOptions:
    return KeyedSearchOptions(maximum_charge=CHARGE, **fields)


def test_absent_deadline_is_omitted_from_the_wire() -> None:
    assert "deadline_seconds" not in _options(mode="deep").model_dump(mode="json")


def test_deep_deadline_up_to_600_serializes() -> None:
    assert _options(mode="deep", deadline_seconds=600).model_dump(mode="json")["deadline_seconds"] == 600


@pytest.mark.parametrize("seconds", [0, 601])
def test_deadline_outside_1_to_600_is_rejected(seconds: int) -> None:
    with pytest.raises(ValidationError):
        _options(mode="deep", deadline_seconds=seconds)


def test_deadline_requires_deep_mode() -> None:
    with pytest.raises(ValidationError):
        _options(mode="fast", deadline_seconds=30)
