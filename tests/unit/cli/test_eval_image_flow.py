import json
from unittest import mock

import pytest

from click.testing import CliRunner
from synth_ai.cli.task_apps import eval_command


@pytest.mark.parametrize("use_remote", [False, True])
def test_eval_multimodal_paths_are_built(monkeypatch, tmp_path, use_remote):
    # Mock ASGI transport and httpx client to avoid any network
    class DummyResp:
        def __init__(self):
            self.status_code = 200
        def json(self):
            # Minimal structured response expected by eval: include trace payload
            return {"status": "ok", "reward": 0, "trace": {"session_trace": {"session_id": "sess", "created_at": "2025-01-01T00:00:00Z", "num_timesteps": 0, "num_events": 0, "num_messages": 0, "metadata": {}}}}

    async def fake_post(*args, **kwargs):
        return DummyResp()

    monkeypatch.setattr("httpx.AsyncClient.post", mock.AsyncMock(side_effect=fake_post))

    # Provide a minimal config via temp file
    cfg = tmp_path / "eval.toml"
    cfg.write_text(
        """
[eval]
app_id = "grpo-crafter-task-app"
model = "gpt-4o-mini-2024-07-18"
seeds = [0]
trace_db = "none"
image_only_mode = true
use_vision = true
""",
        encoding="utf-8",
    )

    # Remote vs in-process toggle
    url = "http://localhost:9999" if use_remote else None

    # Run eval command via Click runner; network calls are mocked
    runner = CliRunner()
    args = ["grpo-crafter-task-app", "--config", str(cfg), "--seeds", "0", "--trace-db", "none"]
    if url:
        args += ["--url", url, "--env-file", str(cfg)]  # any file path satisfies option requirement
    result = runner.invoke(eval_command, args)
    assert result.exit_code == 0, result.output


