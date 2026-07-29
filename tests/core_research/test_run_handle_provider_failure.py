"""Terminal provider failures remain typed at the managed-run SDK boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from synth_ai.core.research.errors import SmrInferenceProviderUnavailableError
from synth_ai.core.research.session.runs import RunHandle


def test_wait_raises_typed_provider_unavailable_error_from_run_contract() -> None:
    contract = SimpleNamespace(
        terminal=True,
        public_state=SimpleNamespace(value="failed"),
        diagnostics=SimpleNamespace(
            failure_classification={
                "code": "inference_provider_unavailable",
                "detail": "Kimi is experiencing temporary high demand.",
                "provider": "modal",
                "route": {"model": "kimi-k3", "provider": "modal"},
                "retryable": True,
                "upstream_status": 503,
            }
        ),
    )
    handle = SimpleNamespace(run_id="run_123", contract=lambda: contract)

    with pytest.raises(SmrInferenceProviderUnavailableError) as excinfo:
        RunHandle.wait(handle, poll_interval=0.1, raise_if_failed=True)

    exc = excinfo.value
    assert exc.retryable is True
    assert exc.provider == "modal"
    assert exc.model == "kimi-k3"
    assert exc.upstream_status == 503
