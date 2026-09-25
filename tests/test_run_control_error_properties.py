"""A refused run lifecycle control is raised, not replaced by AttributeError.

2026-09-23: SynthError exposes error_code and retryable as read-only
properties; the subclass assigned both, so stopping an already-terminal run
crashed CardCode's interrupt cleanup instead of reporting the refusal.
"""

from synth_ai.sdk.research.contracts.run_control import (
    ManagedResearchRunControlError,
    RunLifecycleControlErrorCode,
)


def test_a_run_control_refusal_carries_its_code_and_retryability():
    code = next(iter(RunLifecycleControlErrorCode))
    error = ManagedResearchRunControlError(
        error_code=code,
        message="Run is already terminal (failed)",
        retryable=False,
        current_state="failed",
        run_id="run-1",
    )
    assert error.error_code is code
    assert error.retryable is False
    assert error.current_state == "failed"
