import json
from datetime import datetime, UTC

import pytest
pytest.importorskip("libsql")

from synth_ai.core.tracing_v3.abstractions import LMCAISEvent, SessionTrace, TimeRecord
from synth_ai.core.tracing_v3.turso.native_manager import NativeLibsqlTraceManager


@pytest.mark.asyncio
async def test_cais_call_records_dict_and_dataclass(tmp_path):
    db_path = tmp_path / "traces.db"
    mgr = NativeLibsqlTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")

    # Event with dict call_records
    e1 = LMCAISEvent(
        system_instance_id="llm",
        model_name="mock",
        provider="openai",
        time_record=TimeRecord(event_time=0.0, message_time=0),
        call_records=[{"request": {"messages": [{"role": "user", "content": "hi"}]}, "response": {"content": "ok"}}],
    )

    trace = SessionTrace(session_id="sess_cais_1", created_at=datetime.now(UTC), event_history=[e1])
    await mgr.insert_session_trace(trace)

    rows = await mgr.query_traces("SELECT call_records FROM events WHERE session_id = :sid", {"sid": "sess_cais_1"})
    recs = rows.to_dict("records") if hasattr(rows, "to_dict") else rows
    assert len(recs) == 1
    cr = recs[0]["call_records"]
    if isinstance(cr, str):
        cr = json.loads(cr)
    assert isinstance(cr, list)
    assert isinstance(cr[0], dict)
    assert "request" in cr[0]

