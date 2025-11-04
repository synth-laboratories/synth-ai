from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from synth_ai.tracing_v3.session_tracer import SessionTracer
from synth_ai.tracing_v3.turso.native_manager import NativeLibsqlTraceManager


async def _run_with_factory(factory, session_id: str) -> str:
    tracer: SessionTracer = factory()
    try:
        await tracer.initialize()
        await tracer.start_session(session_id=session_id, metadata={})
        await tracer.record_message(content="hello", message_type="system", metadata={})
        await asyncio.sleep(0.01)
        trace = await tracer.end_session()
        return trace.session_id if trace is not None else "missing"
    finally:
        await tracer.close()


@pytest.mark.asyncio
async def test_session_tracer_factory_supports_parallel_sessions(tmp_path: Path) -> None:
    """Ensure factories return isolated tracers so concurrent rollouts succeed."""

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'trace.db'}"

    def tracer_factory() -> SessionTracer:
        return SessionTracer(storage=NativeLibsqlTraceManager(db_url=db_url))

    results = await asyncio.gather(
        _run_with_factory(tracer_factory, "session-a"),
        _run_with_factory(tracer_factory, "session-b"),
    )

    assert set(results) == {"session-a", "session-b"}
