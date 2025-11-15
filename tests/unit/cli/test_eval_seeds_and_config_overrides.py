from unittest import mock

import pytest
from click.testing import CliRunner
from synth_ai.cli.commands.eval.core import eval_command


@pytest.mark.timeout(30)  # Increase timeout for filesystem scanning
def test_eval_in_process_and_remote_overrides(monkeypatch, tmp_path):
    # Mock httpx client to avoid network
    class DummyResp:
        status_code = 200
        def json(self):
            return {"status": "ok", "trace": {"session_trace": {"session_id": "sess", "created_at": "2025-01-01T00:00:00Z", "num_timesteps": 0, "num_events": 0, "num_messages": 0, "metadata": {}}}}

    monkeypatch.setattr("httpx.AsyncClient.post", mock.AsyncMock(return_value=DummyResp()))

    # Config file with two seeds
    cfg = tmp_path / "eval.toml"
    cfg.write_text(
        """
[eval]
app_id = "grpo-crafter-task-app"
model = "gpt-4o-mini-2024-07-18"
seeds = [1, 2]
trace_db = "none"
""",
        encoding="utf-8",
    )

    env_file = tmp_path / ".env"
    env_file.write_text("SYNTH_API_KEY=test-synth\nENVIRONMENT_API_KEY=test-env\n", encoding="utf-8")

    # Run in-process (url None)
    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            "grpo-crafter-task-app",
            "--config",
            str(cfg),
            "--seeds",
            "1,2",
            "--trace-db",
            "none",
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code == 0, result.output

    # Run remote (url provided) uses same code path but with transport mocked
    result = runner.invoke(
        eval_command,
        [
            "grpo-crafter-task-app",
            "--config",
            str(cfg),
            "--url",
            "http://localhost:9999",
            "--env-file",
            str(cfg),
            "--seeds",
            "1,2",
            "--trace-db",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output

