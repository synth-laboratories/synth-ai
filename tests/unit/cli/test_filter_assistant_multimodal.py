import json
from datetime import datetime, UTC

import pytest
import asyncio

from synth_ai.tracing_v3.abstractions import (
    SessionTrace,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    TimeRecord,
)
from synth_ai.tracing_v3.turso.native_manager import NativeLibsqlTraceManager
from click.testing import CliRunner
from synth_ai.cli.task_apps import filter_command


def test_filter_preserves_assistant_multimodal(tmp_path):
    db_path = tmp_path / "traces.db"
    mgr = NativeLibsqlTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")

    # Build a session with assistant multimodal output
    session_id = "sess_assistant_mm"
    trace = SessionTrace(
        session_id=session_id,
        created_at=datetime.now(UTC),
        markov_blanket_message_history=[
            SessionEventMarkovBlanketMessage(
                message_type="user",
                content=SessionMessageContent(
                    json_payload={
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://img/x.png"}}],
                    }
                ),
                time_record=TimeRecord(event_time=0.0, message_time=0),
                metadata={"step_id": "s0"},
            ),
            SessionEventMarkovBlanketMessage(
                message_type="assistant",
                content=SessionMessageContent(
                    json_payload={
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Here's a description"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                        ],
                    }
                ),
                time_record=TimeRecord(event_time=0.1, message_time=1),
                metadata={"step_id": "s0"},
            ),
        ],
        event_history=[],
        session_time_steps=[],
        metadata={"env_name": "crafter", "policy_name": "crafter-react", "seed": 0, "model": "mock"},
    )

    async def _setup():
        await mgr.insert_session_trace(trace)
    asyncio.run(_setup())

    # Write filter config
    out_path = tmp_path / "out.jsonl"
    cfg_path = tmp_path / "filter.toml"
    cfg_path.write_text(
        f"""
[filter]
db = "sqlite+aiosqlite:///{db_path}"
output = "{out_path}"
""",
        encoding="utf-8",
    )

    # Run filtering via Click runner
    runner = CliRunner()
    result = runner.invoke(filter_command, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # Validate assistant content remains a list (multimodal)
    line = out_path.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assistant = rec["messages"][1]
    assert isinstance(assistant["content"], list)
    assert any(p.get("type") == "image_url" for p in assistant["content"] if isinstance(p, dict))


