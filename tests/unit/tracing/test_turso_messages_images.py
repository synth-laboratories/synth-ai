import asyncio
import json
from datetime import datetime, UTC

import pytest
pytest.importorskip("libsql")

from synth_ai.tracing_v3.abstractions import (
    SessionTrace,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    TimeRecord,
)
from synth_ai.tracing_v3.turso.native_manager import NativeLibsqlTraceManager


@pytest.mark.asyncio
async def test_insert_and_query_messages_with_multimodal_json(tmp_path):
    db_path = tmp_path / "traces.db"
    mgr = NativeLibsqlTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")

    session_id = "sess_vision_1"
    trace = SessionTrace(
        session_id=session_id,
        created_at=datetime.now(UTC),
        markov_blanket_message_history=[
            SessionEventMarkovBlanketMessage(
                message_type="user",
                content=SessionMessageContent(
                    json_payload={
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                        ],
                    }
                ),
                time_record=TimeRecord(event_time=0.0, message_time=0),
            )
        ],
        event_history=[],
        session_time_steps=[],
        metadata={}
    )

    await mgr.insert_session_trace(trace)

    rows = await mgr.query_traces(
        "SELECT content, metadata FROM messages WHERE session_id = :sid",
        {"sid": session_id},
    )

    recs = rows.to_dict("records") if hasattr(rows, "to_dict") else rows
    assert len(recs) == 1

    # Content should be stored as JSON (not huge base64 embedded into prompt text)
    content = recs[0]["content"]
    if isinstance(content, str):
        content = json.loads(content)
    assert isinstance(content, (dict, list))
    # Expect multimodal list present
    parts = content["content"] if isinstance(content, dict) else content
    assert any(isinstance(p, dict) and p.get("type") == "image_url" for p in parts)


