from __future__ import annotations

import pytest

from synth_ai.core.tunnels.synth_tunnel import _request_timeout_from_deadline_ms


def test_request_timeout_uses_deadline_for_total_and_read() -> None:
    timeout = _request_timeout_from_deadline_ms(1500)
    assert timeout.read == pytest.approx(1.5)
    assert timeout.pool == pytest.approx(1.5)
    assert timeout.connect == pytest.approx(1.5)
    assert timeout.write == pytest.approx(1.5)


def test_request_timeout_caps_connect_and_write_for_large_deadlines() -> None:
    timeout = _request_timeout_from_deadline_ms(120_000)
    assert timeout.read == pytest.approx(120.0)
    assert timeout.pool == pytest.approx(120.0)
    assert timeout.connect == pytest.approx(10.0)
    assert timeout.write == pytest.approx(10.0)


def test_request_timeout_has_small_positive_floor() -> None:
    timeout = _request_timeout_from_deadline_ms(0)
    assert timeout.read == pytest.approx(0.05)
    assert timeout.pool == pytest.approx(0.05)
    assert timeout.connect == pytest.approx(0.05)
    assert timeout.write == pytest.approx(0.05)
