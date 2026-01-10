import os
from unittest import mock

import pytest
from click.testing import CliRunner
from synth_ai.cli.claude import claude as claude_cmd
from synth_ai.core.urls import BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_env():
    """Mock environment variables."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/test",
    }


def test_claude_cmd_no_claude_binary_found(runner: CliRunner):
    """Test that claude_cmd exits gracefully when Claude Code is not found."""
    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=None),
        mock.patch("synth_ai.core.agents.claude.install_bin", return_value=False),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to find your installed Claude Code" in result.output
    assert "Please install from: https://claude.com/claude-code" in result.output


def test_claude_cmd_with_default_url(runner: CliRunner, mock_env):
    """Test claude_cmd with default URL (no override)."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-123"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Using Claude at {mock_bin_path}" in result.output

    # Verify subprocess.run was called correctly
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["claude"]
    assert call_args[1]["check"] is True

    # Verify environment variables
    env = call_args[1]["env"]
    assert env["ANTHROPIC_BASE_URL"] == f"{BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC}/synth-small"
    assert env["ANTHROPIC_AUTH_TOKEN"] == mock_api_key
    assert env["SYNTH_API_KEY"] == mock_api_key


def test_claude_cmd_with_override_url(runner: CliRunner, mock_env):
    """Test claude_cmd with custom override URL."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-456"
    override_url = "https://custom.example.com/api"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small", "--url", override_url])

    assert result.exit_code == 0
    assert "Using override URL with model:" in result.output

    # Verify environment variables
    env = mock_run.call_args[1]["env"]
    assert env["ANTHROPIC_BASE_URL"] == f"{override_url}/synth-small"


def test_claude_cmd_with_override_url_trailing_slash(runner: CliRunner, mock_env):
    """Test that override URL trailing slashes are properly handled."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-789"
    override_url = "https://custom.example.com/api/"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small", "--url", override_url])

    assert result.exit_code == 0

    # Verify trailing slash is removed
    env = mock_run.call_args[1]["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://custom.example.com/api/synth-small"


def test_claude_cmd_with_force_flag(runner: CliRunner, mock_env):
    """Test that --force flag is passed to resolve_env_var."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-force"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch(
            "synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key
        ) as mock_resolve,
        mock.patch("synth_ai.core.agents.claude.subprocess.run"),
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small", "--force"])

    assert result.exit_code == 0

    # Verify resolve_env_var was called with force=True
    mock_resolve.assert_called_once_with("SYNTH_API_KEY", override_process_env=True)


def test_claude_cmd_subprocess_error(runner: CliRunner, mock_env):
    """Test that subprocess errors are handled gracefully."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-error"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        # Simulate subprocess failure
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "claude")

        result = runner.invoke(claude_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to launch Claude Code" in result.output


def test_claude_cmd_different_models(runner: CliRunner, mock_env):
    """Test claude_cmd with different model names."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-models"

    models = ["synth-small", "synth-medium"]

    for model in models:
        with (
            mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
            mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
            mock.patch("synth_ai.core.agents.claude.write_agents_md"),
            mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
            mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
            mock.patch.dict(os.environ, mock_env, clear=True),
        ):
            result = runner.invoke(claude_cmd, ["--model", model])

        assert result.exit_code == 0

        # Verify model name is in URL
        env = mock_run.call_args[1]["env"]
        assert env["ANTHROPIC_BASE_URL"].endswith(f"/{model}")


def test_claude_cmd_preserves_existing_env_vars(runner: CliRunner):
    """Test that claude_cmd preserves existing environment variables."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-preserve"

    test_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/test",
        "CUSTOM_VAR": "custom_value",
        "ANOTHER_VAR": "another_value",
    }

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", return_value=mock_bin_path),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run") as mock_run,
        mock.patch.dict(os.environ, test_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0

    # Verify all original env vars are preserved
    env = mock_run.call_args[1]["env"]
    assert env["CUSTOM_VAR"] == "custom_value"
    assert env["ANOTHER_VAR"] == "another_value"
    assert env["PATH"] == "/usr/bin:/bin"


def test_claude_cmd_install_loop_success(runner: CliRunner, mock_env):
    """Test that claude_cmd retries installation if not found initially."""
    mock_bin_path = "/usr/local/bin/claude"
    mock_api_key = "test-api-key-install"

    with (
        mock.patch("synth_ai.core.agents.claude.get_bin_path", side_effect=[None, mock_bin_path]),
        mock.patch("synth_ai.core.agents.claude.install_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.verify_bin", return_value=True),
        mock.patch("synth_ai.core.agents.claude.write_agents_md"),
        mock.patch("synth_ai.core.agents.claude.resolve_env_var", return_value=mock_api_key),
        mock.patch("synth_ai.core.agents.claude.subprocess.run"),
        mock.patch.dict(os.environ, mock_env, clear=True),
    ):
        result = runner.invoke(claude_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Using Claude at {mock_bin_path}" in result.output
