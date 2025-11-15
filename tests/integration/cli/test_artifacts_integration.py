"""Integration tests for artifacts CLI commands.

These tests require a running backend and valid API credentials.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Skip if backend URL or API key not configured
BACKEND_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("SYNTH_API_KEY") or os.getenv("TESTING_LOCAL_SYNTH_API_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY or BACKEND_URL == "http://localhost:8000",
    reason="Requires backend URL and API key",
)


def _run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run CLI command and return result."""
    cmd = ["uvx", "synth-ai"] + args
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, text=True, capture_output=True, env=full_env)


class TestArtifactsListIntegration:
    """Integration tests for artifacts list command."""

    def test_list_all_artifacts(self) -> None:
        """Test listing all artifacts."""
        result = _run_cli(
            ["artifacts", "list"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Summary" in result.stdout or "summary" in result.stdout.lower()

    def test_list_models(self) -> None:
        """Test listing models only."""
        result = _run_cli(
            ["artifacts", "list", "--type", "models"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_list_prompts(self) -> None:
        """Test listing prompts only."""
        result = _run_cli(
            ["artifacts", "list", "--type", "prompts"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_list_json_format(self) -> None:
        """Test listing artifacts in JSON format."""
        result = _run_cli(
            ["artifacts", "list", "--format", "json"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        import json

        # Should be valid JSON
        data = json.loads(result.stdout)
        assert "summary" in data or "models" in data or "prompts" in data


class TestArtifactsShowIntegration:
    """Integration tests for artifacts show command."""

    @pytest.mark.skip(reason="Requires existing prompt job ID")
    def test_show_prompt_default(self) -> None:
        """Test showing prompt details in default mode."""
        # Replace with actual prompt job ID from your org
        prompt_id = "pl_71c12c4c7c474c34"
        result = _run_cli(
            ["artifacts", "show", prompt_id],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert prompt_id in result.stdout

    @pytest.mark.skip(reason="Requires existing prompt job ID")
    def test_show_prompt_verbose(self) -> None:
        """Test showing prompt details in verbose mode."""
        prompt_id = "pl_71c12c4c7c474c34"
        result = _run_cli(
            ["artifacts", "show", prompt_id, "--verbose"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Full Details" in result.stdout or "Metadata" in result.stdout

    @pytest.mark.skip(reason="Requires existing prompt job ID")
    def test_show_prompt_json(self) -> None:
        """Test showing prompt details in JSON format."""
        prompt_id = "pl_71c12c4c7c474c34"
        result = _run_cli(
            ["artifacts", "show", prompt_id, "--format", "json"],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        import json

        data = json.loads(result.stdout)
        assert "job_id" in data
        assert data["job_id"] == prompt_id

    @pytest.mark.skip(reason="Requires existing model ID")
    def test_show_model(self) -> None:
        """Test showing model details."""
        # Replace with actual model ID from your org
        model_id = "ft:Qwen/Qwen3-0.6B:job_12345"
        result = _run_cli(
            ["artifacts", "show", model_id],
            env={"SYNTH_API_KEY": API_KEY, "BACKEND_BASE_URL": BACKEND_URL},
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"


class TestArtifactsCommandRegistration:
    """Test that artifacts commands are properly registered."""

    def test_artifacts_command_exists(self) -> None:
        """Test that artifacts command is registered."""
        result = _run_cli(["artifacts", "--help"])
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "list" in result.stdout or "show" in result.stdout

    def test_artifacts_list_help(self) -> None:
        """Test artifacts list help."""
        result = _run_cli(["artifacts", "list", "--help"])
        assert result.returncode == 0, f"Command failed: {result.stderr}"

    def test_artifacts_show_help(self) -> None:
        """Test artifacts show help."""
        result = _run_cli(["artifacts", "show", "--help"])
        assert result.returncode == 0, f"Command failed: {result.stderr}"

