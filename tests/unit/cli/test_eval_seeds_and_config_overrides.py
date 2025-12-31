from unittest import mock

import pytest
from click.testing import CliRunner
from synth_ai.cli.commands.eval.core import eval_command


@pytest.mark.slow
def test_eval_in_process_and_remote_overrides(monkeypatch, tmp_path):
    # Mock httpx client to avoid network
    class DummyResp:
        status_code = 200
        def json(self):
            return {
                "run_id": "test",
                "trajectories": [
                    {
                        "env_id": "env",
                        "policy_id": "policy",
                        "steps": [
                            {
                                "obs": {},
                                "tool_calls": [],
                                "reward": 0.0,
                                "done": True,
                                "info": {"messages": [{"role": "user", "content": "hi"}]},
                            }
                        ],
                        "final": {},
                        "length": 1,
                        "inference_url": "http://localhost",
                    }
                ],
                "branches": {},
                "metrics": {
                    "episode_rewards": [0.0],
                    "reward_mean": 0.0,
                    "num_steps": 1,
                    "num_episodes": 1,
                },
                "aborted": False,
                "trace": {"session_id": "sess", "event_history": [], "metadata": {}},
            }

    monkeypatch.setattr("httpx.AsyncClient.request", mock.AsyncMock(return_value=DummyResp()))

    # Config file with two seeds
    cfg = tmp_path / "eval.toml"
    cfg.write_text(
        """
[eval]
app_id = "banking77"
model = "gpt-4o-mini-2024-07-18"
seeds = [1, 2]
trace_db = "none"
task_app_url = "http://localhost:9999"
""",
        encoding="utf-8",
    )

    env_file = tmp_path / ".env"
    env_file.write_text("SYNTH_API_KEY=test-synth\nENVIRONMENT_API_KEY=test-env\n", encoding="utf-8")

    # Run with task_app_url from config
    runner = CliRunner()
    result = runner.invoke(
        eval_command,
        [
            "banking77",
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
            "banking77",
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
