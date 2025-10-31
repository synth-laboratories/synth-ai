from unittest import mock

from click.testing import CliRunner
from synth_ai.cli.task_apps import eval_command


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
    env_file.write_text("ENVIRONMENT_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-key")

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
            str(env_file),
            "--seeds",
            "1,2",
            "--trace-db",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output

