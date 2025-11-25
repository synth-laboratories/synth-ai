from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from click.testing import CliRunner
from synth_ai.cli.commands.filter.core import filter_command
from synth_ai.core.tracing_v3.abstractions import (
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    SessionTrace,
    TimeRecord,
)
from synth_ai.core.tracing_v3.turso.native_manager import NativeLibsqlTraceManager


def _build_trace(
    *,
    session_id: str,
    created_at: datetime,
    split: str,
    task_id: str,
    model: str,
    judge_scores: dict[str, float] | None = None,
    metadata_extra: dict | None = None,
    user_text: str = "user message",
    assistant_text: str = "assistant reply",
) -> SessionTrace:
    metadata = {
        "task_split": split,
        "task_id": task_id,
        "model": model,
        "env_name": "demo",
        "policy_name": "demo-policy",
        "seed": 0,
    }
    if judge_scores:
        metadata["judge_scores"] = judge_scores
    if metadata_extra:
        metadata.update(metadata_extra)

    messages = [
        SessionEventMarkovBlanketMessage(
            message_type="user",
            content=SessionMessageContent(
                json_payload={"role": "user", "content": user_text}
            ),
            time_record=TimeRecord(event_time=0.0, message_time=0),
            metadata={},
        ),
        SessionEventMarkovBlanketMessage(
            message_type="assistant",
            content=SessionMessageContent(
                json_payload={"role": "assistant", "content": assistant_text}
            ),
            time_record=TimeRecord(event_time=0.1, message_time=1),
            metadata={},
        ),
    ]

    return SessionTrace(
        session_id=session_id,
        created_at=created_at,
        markov_blanket_message_history=messages,
        event_history=[],
        session_time_steps=[],
        metadata=metadata,
    )


def _write_sessions(
    mgr: NativeLibsqlTraceManager,
    sessions: list[SessionTrace],
    rewards: dict[str, tuple[int, int]] | None = None,
) -> None:
    async def _inner() -> None:
        for trace in sessions:
            await mgr.insert_session_trace(trace)
            if rewards and trace.session_id in rewards:
                total_reward, achievements = rewards[trace.session_id]
                await mgr.insert_outcome_reward(
                    trace.session_id,
                    total_reward=total_reward,
                    achievements_count=achievements,
                    total_steps=1,
                )

    asyncio.run(_inner())


def _run_filter(
    tmp_path: Path,
    *,
    sessions: list[SessionTrace],
    config_body: str,
    rewards: dict[str, tuple[int, int]] | None = None,
) -> tuple[int, list[dict[str, object]], str]:
    db_path = tmp_path / "traces.db"
    mgr = NativeLibsqlTraceManager(db_url=f"sqlite+aiosqlite:///{db_path}")
    _write_sessions(mgr, sessions, rewards)

    output_path = tmp_path / "out.jsonl"
    config_text = config_body.format(
        db_url=f"sqlite+aiosqlite:///{db_path}",
        output_path=str(output_path).replace("\\", "\\\\"),
    )

    config_path = tmp_path / "filter.toml"
    config_path.write_text(config_text, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(filter_command, ["--config", str(config_path)])

    records: list[dict[str, object]] = []
    if output_path.exists() and output_path.read_text(encoding="utf-8").strip():
        records = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").strip().splitlines()
        ]

    return result.exit_code, records, result.output


def _session_ids(records: list[dict[str, object]]) -> list[str]:
    return [rec["metadata"]["session_id"] for rec in records]


def test_filter_by_split_and_task(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions = [
        _build_trace(
            session_id="train-1",
            created_at=now,
            split="train",
            task_id="task-a",
            model="model-a",
        ),
        _build_trace(
            session_id="test-1",
            created_at=now,
            split="test",
            task_id="task-b",
            model="model-b",
        ),
    ]

    config_body = """
[filter]
db = "{db_url}"
output = "{output_path}"
splits = ["train"]
task_ids = ["task-a"]
"""

    exit_code, records, _ = _run_filter(
        tmp_path,
        sessions=sessions,
        config_body=config_body,
    )

    assert exit_code == 0
    assert _session_ids(records) == ["train-1"]


def test_filter_by_model_and_official_score(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions = [
        _build_trace(
            session_id="session-low",
            created_at=now,
            split="train",
            task_id="task-a",
            model="model-a",
        ),
        _build_trace(
            session_id="session-high",
            created_at=now,
            split="train",
            task_id="task-a",
            model="model-b",
        ),
    ]

    rewards = {
        "session-low": (3, 0),
        "session-high": (9, 1),
    }

    config_body = """
[filter]
db = "{db_url}"
output = "{output_path}"
models = ["model-b"]
min_official_score = 5.0
"""

    exit_code, records, _ = _run_filter(
        tmp_path,
        sessions=sessions,
        rewards=rewards,
        config_body=config_body,
    )

    assert exit_code == 0
    assert _session_ids(records) == ["session-high"]


def test_filter_by_judge_scores(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions = [
        _build_trace(
            session_id="good",
            created_at=now,
            split="train",
            task_id="task",
            model="model",
            judge_scores={"quality": 0.9},
        ),
        _build_trace(
            session_id="poor",
            created_at=now,
            split="train",
            task_id="task",
            model="model",
            judge_scores={"quality": 0.5},
        ),
    ]

    config_body = """
[filter]
db = "{db_url}"
output = "{output_path}"
[filter.min_judge_scores]
quality = 0.8
"""

    exit_code, records, _ = _run_filter(
        tmp_path,
        sessions=sessions,
        config_body=config_body,
    )

    assert exit_code == 0
    assert _session_ids(records) == ["good"]


def test_filter_by_created_at_and_limit(tmp_path: Path) -> None:
    base = datetime.now(UTC)
    sessions = [
        _build_trace(
            session_id="old",
            created_at=base - timedelta(days=2),
            split="train",
            task_id="task",
            model="m",
        ),
        _build_trace(
            session_id="recent",
            created_at=base - timedelta(hours=1),
            split="train",
            task_id="task",
            model="m",
        ),
        _build_trace(
            session_id="latest",
            created_at=base,
            split="train",
            task_id="task",
            model="m",
        ),
    ]

    min_created = (base - timedelta(days=1)).isoformat()

    config_body = f"""
[filter]
db = "{{db_url}}"
output = "{{output_path}}"
min_created_at = "{min_created}"
limit = 1
"""

    exit_code, records, _ = _run_filter(
        tmp_path,
        sessions=sessions,
        config_body=config_body,
    )

    assert exit_code == 0
    # Records ordered by created_at ascending
    assert _session_ids(records) == ["recent"]


def test_filter_by_max_official_and_max_judge(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions = [
        _build_trace(
            session_id="ok",
            created_at=now,
            split="train",
            task_id="task",
            model="m",
            judge_scores={"toxicity": 0.1},
        ),
        _build_trace(
            session_id="bad",
            created_at=now,
            split="train",
            task_id="task",
            model="m",
            judge_scores={"toxicity": 0.7},
        ),
    ]

    rewards = {
        "ok": (1, 0),
        "bad": (9, 0),
    }

    config_body = """
[filter]
db = "{db_url}"
output = "{output_path}"
max_official_score = 5.0
[filter.max_judge_scores]
toxicity = 0.2
"""

    exit_code, records, _ = _run_filter(
        tmp_path,
        sessions=sessions,
        rewards=rewards,
        config_body=config_body,
    )

    assert exit_code == 0
    assert _session_ids(records) == ["ok"]
