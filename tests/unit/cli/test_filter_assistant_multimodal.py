import json
from datetime import datetime, UTC
from typing import Literal

import click
import pytest
import asyncio

from synth_ai.core.tracing_v3.abstractions import (
    SessionTrace,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    TimeRecord,
)
from synth_ai.core.tracing_v3.turso.native_manager import NativeLibsqlTraceManager
from click.testing import CliRunner

from synth_ai.cli.commands.filter import core as filter_core
from synth_ai.cli.commands.filter.core import filter_command
from synth_ai.cli.commands.filter.errors import (
    FilterConfigNotFoundError,
    FilterConfigParseError,
    InvalidFilterConfigError,
    MissingFilterTableError,
    NoSessionsMatchedError,
    NoTracesFoundError,
    TomlUnavailableError,
)
from synth_ai.sdk.task.config import FilterConfig


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


def _invoke_filter_with_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    *,
    phase: Literal["load", "run"],
) -> click.testing.Result:
    runner = CliRunner()
    if phase == "load":
        monkeypatch.setattr(
            filter_core,
            "_load_filter_config",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )
    else:
        config = FilterConfig.from_dict(
            {
                "db": "sqlite+aiosqlite:///dummy.db",
                "output": "out.jsonl",
            }
        )
        monkeypatch.setattr(
            filter_core,
            "_load_filter_config",
            lambda *_args, **_kwargs: (config, {}),
        )
        monkeypatch.setattr(
            filter_core.asyncio,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )
    return runner.invoke(filter_command, ["--config", "dummy.toml"])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TomlUnavailableError(hint="Install tomli"), "TOML parser not available"),
        (FilterConfigNotFoundError(path="missing.toml"), "Filter config not found: missing.toml"),
        (
            FilterConfigParseError(path="bad.toml", detail="boom"),
            "Failed to parse TOML 'bad.toml': boom",
        ),
        (MissingFilterTableError(), "Config must contain a [filter] table."),
        (
            InvalidFilterConfigError(detail="missing db"),
            "Invalid filter config: missing db",
        ),
    ],
)
def test_filter_formats_load_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    result = _invoke_filter_with_error(monkeypatch, error, phase="load")
    assert result.exit_code != 0
    assert expected in result.output


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NoTracesFoundError(db_url="sqlite+aiosqlite:///dummy.db"), "No traces found in database"),
        (NoSessionsMatchedError(), "No sessions matched the provided filters"),
    ],
)
def test_filter_formats_runtime_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    result = _invoke_filter_with_error(monkeypatch, error, phase="run")
    assert result.exit_code != 0
    assert expected in result.output
