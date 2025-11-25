import asyncio
import json
from datetime import datetime, UTC
from types import SimpleNamespace

import pytest

from synth_ai.core.tracing_v3.session_tracer import SessionTracer
from synth_ai.core.tracing_v3.turso.native_manager import NativeLibsqlTraceManager
from click.testing import CliRunner
from synth_ai.cli.task_apps import filter_command


def _write_toml(tmp_path, db_path, out_path):
    toml_text = f"""
[filter]
db = "sqlite+aiosqlite:///{db_path}"
output = "{out_path}"
limit = 10
"""
    cfg = tmp_path / "filter.toml"
    cfg.write_text(toml_text, encoding="utf-8")
    return cfg


def test_filter_preserves_multimodal_messages(tmp_path, monkeypatch):
    db_path = tmp_path / "traces.db"
    mgr = NativeLibsqlTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")

    # Insert minimal session with a multimodal user message
    from synth_ai.core.tracing_v3.abstractions import (
        SessionTrace,
        SessionEventMarkovBlanketMessage,
        SessionMessageContent,
        TimeRecord,
    )

    session_id = "sess_filter_1"
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
                            {"type": "image_url", "image_url": {"url": "https://ok/a.png"}},
                            {"type": "text", "text": "what is this?"},
                        ],
                    }
                ),
                time_record=TimeRecord(event_time=0.0, message_time=0),
                metadata={"step_id": "s0"},
            ),
            SessionEventMarkovBlanketMessage(
                message_type="assistant",
                content=SessionMessageContent(text="It looks like a tree."),
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

    # Monkeypatch tracer in filter_command to use our DB
    out_path = tmp_path / "out.jsonl"
    cfg_path = _write_toml(tmp_path, db_path, out_path)

    # Run filter_command via Click runner
    runner = CliRunner()
    result = runner.invoke(filter_command, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # Validate output contains list content for user message
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    user = rec["messages"][0]
    assert isinstance(user["content"], list), "user content should preserve multimodal list"
    assert any(p.get("type") == "image_url" for p in user["content"] if isinstance(p, dict))


