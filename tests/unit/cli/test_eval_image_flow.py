import json
from unittest import mock

import pytest

from click.testing import CliRunner
from synth_ai.cli.commands.eval.core import eval_command


@pytest.mark.slow
@pytest.mark.parametrize("use_remote", [False, True])
def test_eval_multimodal_paths_are_built(monkeypatch, tmp_path, use_remote):
    # Mock ASGI transport and httpx client to avoid any network
    class DummyResp:
        def __init__(self):
            self.status_code = 200
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
                "metrics": {
                    "outcome_reward": 0.0,
                },
                "trace": {"session_id": "sess", "event_history": [], "metadata": {}},
            }

    async def fake_post(*args, **kwargs):
        return DummyResp()

    monkeypatch.setattr("httpx.AsyncClient.request", mock.AsyncMock(side_effect=fake_post))

    # Provide a minimal config via temp file
    # Use "banking77" which is available in synth-ai demos
    cfg = tmp_path / "eval.toml"
    cfg.write_text(
        """
[eval]
app_id = "banking77"
model = "gpt-4o-mini-2024-07-18"
seeds = [0]
trace_db = "none"
image_only_mode = true
use_vision = true
task_app_url = "http://localhost:9999"
""",
        encoding="utf-8",
    )

    # Remote vs in-process toggle
    url = "http://localhost:9999" if use_remote else None

    env_file = tmp_path / ".env"
    env_file.write_text("SYNTH_API_KEY=test-synth\nENVIRONMENT_API_KEY=test-env\n", encoding="utf-8")

    # Run eval command via Click runner; network calls are mocked
    runner = CliRunner()
    args = [
        "banking77",
        "--config",
        str(cfg),
        "--seeds",
        "0",
        "--trace-db",
        "none",
        "--env-file",
        str(env_file),
    ]
    if url:
        args += ["--url", url]
    result = runner.invoke(eval_command, args)
    assert result.exit_code == 0, result.output
