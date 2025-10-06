import asyncio
import pytest
import time
from typing import Any, Dict
from unittest.mock import patch


class _FakeJobHandle:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def poll_until_terminal(
        self,
        interval_seconds: float = 2.0,
        max_seconds: float | None = None,
        empty_polls_threshold: int = 5,
        startup_deadline_s: int = 45,
        on_event: Any | None = None,
        on_metric: Any | None = None,
    ) -> Dict[str, Any]:
        # Simulate a long-running poll that would stall if not cancelled
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            # When cancelled, emulate a graceful exit that the caller ignores anyway
            raise
        return {"status": "succeeded"}


async def _fake_stream_job_events(_base: str, _key: str, _job_id: str, *, seconds: int, on_event: Any) -> None:
    # Emit a terminal event quickly, then return
    await asyncio.sleep(0.01)
    evt = {"seq": 999, "type": "workflow.completed", "message": "test terminal"}
    try:
        on_event(evt)
    except Exception:
        pass
    # Optionally linger a tiny bit to simulate stream cleanup
    await asyncio.sleep(0.01)


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_poller_fast_exits_on_terminal_event(monkeypatch: Any) -> None:
    import sys
    # Import the module under test directly via file path to avoid package resolution issues
    import importlib.util as _ilu, os as _os
    _mod_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "run_rl_job.py"))
    spec = _ilu.spec_from_file_location("run_rl_job", _mod_path)
    assert spec is not None and spec.loader is not None, "failed to load run_rl_job spec"
    run_rl_job = _ilu.module_from_spec(spec)
    spec.loader.exec_module(run_rl_job)

    # Patch network-dependent helpers
    monkeypatch.setattr(run_rl_job, "backend_health", _noop)
    monkeypatch.setattr(run_rl_job, "task_app_health", _noop)
    # Patch SSE and JobHandle
    monkeypatch.setattr(run_rl_job, "stream_job_events", _fake_stream_job_events)
    monkeypatch.setattr(run_rl_job, "JobHandle", _FakeJobHandle)

    # Simulate CLI args
    argv_backup = list(sys.argv)
    sys.argv = [
        "run_rl_job.py",
        "--backend-url",
        "http://localhost:8000",
        "--api-key",
        "test_key",
        "--job-id",
        "job_test",
        "--stream-seconds",
        "0",
        "--empty-polls",
        "1",
        "--startup-deadline-s",
        "1",
    ]

    try:
        t0 = time.time()
        rc = await run_rl_job._main()
        elapsed = time.time() - t0
    finally:
        sys.argv = argv_backup

    # Should succeed and return quickly (< 1s) due to terminal SSE event
    assert rc == 0, f"expected rc=0, got {rc}"
    assert elapsed < 1.0, f"poller did not fast-exit, elapsed={elapsed:.2f}s"


