from __future__ import annotations

import importlib
from typing import Any

import pytest
from click.testing import CliRunner
from synth_ai.cli.commands.status.errors import StatusAPIError
from synth_ai.cli.commands.status.subcommands.files import files_group
from synth_ai.cli.commands.status.subcommands.jobs import jobs_group
from synth_ai.cli.commands.status.subcommands.models import models_group
from synth_ai.cli.commands.status.subcommands.runs import runs_group
from synth_ai.cli.commands.status.subcommands.summary import summary_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()



def _patch_status_client(monkeypatch, target_path: str, methods: dict[str, Any]):
    calls: dict[str, Any] = {}

    class StubStatusClient:
        def __init__(self, config):
            calls["config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    for name, return_value in methods.items():
        async def _method(self, *args, _rv=return_value, _name=name, **kwargs):
            calls[_name] = {"args": args, "kwargs": kwargs}
            if callable(_rv):
                return await _rv(*args, **kwargs)  # pragma: no cover - dynamic path
            return _rv

        setattr(StubStatusClient, name, _method)

    module = importlib.import_module(target_path)
    monkeypatch.setattr(module, "StatusAPIClient", StubStatusClient, raising=False)
    return calls


def test_jobs_list_json(monkeypatch, runner: CliRunner):
    calls = _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.jobs",
        {
            "list_jobs": [
                {"job_id": "job_123", "status": "running", "training_type": "rl_online"},
            ]
        },
    )

    result = runner.invoke(jobs_group, ["list", "--json"])

    assert result.exit_code == 0
    assert "\"job_123\"" in result.output
    assert calls["list_jobs"]["kwargs"]["status"] is None
    assert calls["config"].base_url


def test_jobs_logs_follow_once(monkeypatch, runner: CliRunner):
    events = [
        {"id": "evt1", "timestamp": "2025-01-01T00:00:00Z", "message": "first"},
        {"id": "evt2", "timestamp": "2025-01-01T00:01:00Z", "message": "second"},
    ]
    calls = _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.jobs",
        {"get_job_events": events},
    )

    result = runner.invoke(
        jobs_group,
        ["logs", "job_abc", "--json", "--tail", "2"],
    )

    assert result.exit_code == 0
    assert "\"evt1\"" in result.output and "\"evt2\"" in result.output
    assert calls["get_job_events"]["kwargs"]["limit"] == 2


def test_files_get(monkeypatch, runner: CliRunner):
    file_payload = {"id": "file-1", "filename": "dataset.jsonl", "purpose": "fine-tune"}
    _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.files",
        {"get_file": file_payload},
    )

    result = runner.invoke(files_group, ["get", "file-1", "--json"])

    assert result.exit_code == 0
    assert "\"dataset.jsonl\"" in result.output


def test_models_list(monkeypatch, runner: CliRunner):
    models = [{"id": "model-1", "base_model": "Qwen/Qwen3-4B"}]
    calls = _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.models",
        {"list_models": models},
    )

    result = runner.invoke(models_group, ["list", "--limit", "1", "--type", "rl", "--json"])

    assert result.exit_code == 0
    assert "\"model-1\"" in result.output
    assert calls["list_models"]["kwargs"]["model_type"] == "rl"


def test_runs_list(monkeypatch, runner: CliRunner):
    runs = [{"id": "run-1", "status": "succeeded"}]
    _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.runs",
        {"list_job_runs": runs},
    )

    result = runner.invoke(runs_group, ["list", "job123", "--json"])

    assert result.exit_code == 0
    assert "\"run-1\"" in result.output


def test_summary_handles_backend_errors(monkeypatch, runner: CliRunner):
    async def raise_error(*args, **kwargs):
        raise StatusAPIError("fail")

    async def list_jobs(*args, **kwargs):
        return [{"job_id": "job"}]

    calls = _patch_status_client(
        monkeypatch,
        "synth_ai.cli.commands.status.subcommands.summary",
        {
            "list_jobs": list_jobs,
            "list_models": raise_error,
            "list_files": raise_error,
        },
    )

    result = runner.invoke(summary_command, [])

    assert result.exit_code == 0
    assert "job" in result.output
    assert "Training Jobs" in result.output
    assert calls["list_jobs"]["args"] == ()
