from __future__ import annotations

import os
from unittest import mock

import pytest
from click.testing import CliRunner

from synth_ai.cli.codex import codex_cmd
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_OPENAI


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


def test_codex_cmd_codex_not_found_no_install(runner: CliRunner):
    """Test that codex_cmd exits when Codex is not found and install fails."""
    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=None), \
         mock.patch("synth_ai.cli.codex.install_bin", return_value=False):

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to find your installed Codex" in result.output


def test_codex_cmd_codex_found_but_not_runnable(runner: CliRunner):
    """Test that codex_cmd exits when Codex is found but not runnable."""
    mock_bin_path = "/usr/local/bin/codex"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=False):

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Using Codex at {mock_bin_path}" in result.output
    assert "Failed to verify Codex is runnable" in result.output


def test_codex_cmd_with_default_url(runner: CliRunner, mock_env):
    """Test codex_cmd with default URL (no override)."""
    mock_bin_path = "/usr/local/bin/codex"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd)

    assert result.exit_code == 0
    assert f"Using Codex at {mock_bin_path}" in result.output

    # Verify subprocess.run was called correctly
    mock_run.assert_called_once()
    call_args = mock_run.call_args

    # Check command structure
    cmd = call_args[0][0]
    assert cmd[0] == "codex"
    assert "-m" not in cmd

    # No overrides or env mutations should occur (except SYNTH_SESSION_ID which is always added)
    assert "-c" not in cmd
    env = call_args[1]["env"]
    # SYNTH_SESSION_ID is always added by codex_cmd
    assert "SYNTH_SESSION_ID" in env
    # All original env vars should be preserved
    for key, value in mock_env.items():
        assert env[key] == value


def test_codex_cmd_with_override_url(runner: CliRunner, mock_env):
    """Test codex_cmd with custom override URL."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-456"
    override_url = "https://custom.example.com/api"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", "synth-small", "--url", override_url])

    assert result.exit_code == 0
    assert "Using override URL:" in result.output
    assert override_url in result.output

    # Verify override URL is in config
    cmd = mock_run.call_args[0][0]
    provider_config_idx = cmd.index("-c") + 1
    assert override_url in cmd[provider_config_idx]


def test_codex_cmd_with_force_flag(runner: CliRunner, mock_env):
    """Test that --force flag is passed to resolve_env_var."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-force"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key) as mock_resolve, \
         mock.patch("synth_ai.cli.codex.subprocess.run"), \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", "synth-small", "--force"])

    assert result.exit_code == 0

    # Resolve env var is invoked for both session creation and provider config
    assert mock_resolve.call_count == 2
    mock_resolve.assert_has_calls(
        [
            mock.call("SYNTH_API_KEY", override_process_env=True),
            mock.call("SYNTH_API_KEY", override_process_env=True),
        ]
    )


def test_codex_cmd_subprocess_error(runner: CliRunner, mock_env):
    """Test that subprocess errors are handled gracefully."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-error"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
         mock.patch.dict(os.environ, mock_env, clear=True):

        # Simulate subprocess failure
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, "codex")

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to run Codex" in result.output


def test_codex_cmd_install_loop_success(runner: CliRunner, mock_env):
    """Test that codex_cmd retries installation if not found initially."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-install"

    # First call returns None, second call returns path
    find_calls = [None, mock_bin_path]

    with mock.patch("synth_ai.cli.codex.get_bin_path", side_effect=find_calls), \
         mock.patch("synth_ai.cli.codex.install_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run"), \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Using Codex at {mock_bin_path}" in result.output


def test_codex_cmd_config_structure(runner: CliRunner, mock_env):
    """Test that codex_cmd generates correct config overrides."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-config"
    model = "synth-small"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", model])

    assert result.exit_code == 0

    # Verify command structure
    cmd = mock_run.call_args[0][0]

    # Should contain model flag
    assert "-m" in cmd
    model_idx = cmd.index("-m") + 1
    assert cmd[model_idx] == model

    # Should contain multiple -c flags for config overrides
    c_count = cmd.count("-c")
    assert c_count == 3  # Three config overrides

    # Verify config content includes provider, model_provider, and default_model
    config_str = " ".join(cmd)
    assert "model_providers.synth=" in config_str
    assert 'model_provider="synth"' in config_str
    assert f'default_model="{model}"' in config_str


def test_codex_cmd_different_models(runner: CliRunner, mock_env):
    """Test codex_cmd with different model names."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-models"

    models = ["synth-small", "synth-medium"]

    for model in models:
        with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
             mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
             mock.patch("synth_ai.cli.codex.write_agents_md"), \
             mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
             mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
             mock.patch.dict(os.environ, mock_env, clear=True):

            result = runner.invoke(codex_cmd, ["--model", model])

        assert result.exit_code == 0

        # Verify model is in command
        cmd = mock_run.call_args[0][0]
        assert model in cmd


def test_codex_cmd_prints_launch_command(runner: CliRunner, mock_env):
    """Test that codex_cmd prints the launch command."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-launch"

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run"), \
         mock.patch.dict(os.environ, mock_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "codex -m synth-small" in result.output


def test_codex_cmd_preserves_existing_env_vars(runner: CliRunner):
    """Test that codex_cmd preserves existing environment variables."""
    mock_bin_path = "/usr/local/bin/codex"
    mock_api_key = "test-api-key-preserve"

    test_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/test",
        "CUSTOM_VAR": "custom_value",
    }

    with mock.patch("synth_ai.cli.codex.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.codex.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.codex.write_agents_md"), \
         mock.patch("synth_ai.cli.codex.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.codex.subprocess.run") as mock_run, \
         mock.patch.dict(os.environ, test_env, clear=True):

        result = runner.invoke(codex_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0

    # Verify all original env vars are preserved
    env = mock_run.call_args[1]["env"]
    assert env["CUSTOM_VAR"] == "custom_value"
    assert env["PATH"] == "/usr/bin:/bin"
