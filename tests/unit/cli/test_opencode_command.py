from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from synth_ai.cli.opencode import (
    opencode_cmd,
    DIV_START,
    DIV_END,
    CONFIG_PATH,
    AUTH_PATH,
    SYNTH_PROVIDER_ID,
    _ensure_synth_provider_in_config,
    _ensure_synth_api_key_in_auth_file,
)
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_BASE


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


@pytest.fixture()
def empty_config():
    """Return an empty config dict."""
    return {}


@pytest.fixture()
def sample_config():
    """Return a sample config dict."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "other_provider": {
                "name": "Other Provider",
                "models": {},
            }
        },
    }


def test_opencode_cmd_not_found_no_install(runner: CliRunner):
    """Test that opencode_cmd exits when OpenCode is not found and install fails."""
    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=None), \
         mock.patch("synth_ai.cli.opencode.install_opencode", return_value=False):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to find your installed OpenCode" in result.output
    assert DIV_END in result.output


def test_opencode_cmd_found_but_not_runnable(runner: CliRunner):
    """Test that opencode_cmd exits when OpenCode is found but not runnable."""
    mock_bin_path = "/usr/local/bin/opencode"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=False):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Found your installed OpenCode at {mock_bin_path}" in result.output
    assert "Failed to verify your installed OpenCode is runnable" in result.output
    assert DIV_END in result.output


def test_opencode_cmd_with_default_url(runner: CliRunner, mock_env, empty_config):
    """Test opencode_cmd with default URL (no override)."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-123"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run") as mock_run:

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Verified your installed OpenCode is runnable" in result.output
    assert "Registering your Synth API key with OpenCode" in result.output
    assert "Launching OpenCode" in result.output

    # Verify subprocess.run was called correctly
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == [str(mock_bin_path)]
    assert call_args[1]["check"] is True

    # Verify config was written
    assert mock_write.call_count >= 2  # Auth file + config file


def test_opencode_cmd_with_override_url(runner: CliRunner, mock_env, empty_config):
    """Test opencode_cmd with custom override URL."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-456"
    override_url = "https://custom.example.com/api"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small", "--url", override_url])

    assert result.exit_code == 0
    assert "Using override URL:" in result.output
    assert override_url in result.output

    # Verify override URL was written to config
    config_write_call = [call for call in mock_write.call_args_list if call[0][0] == CONFIG_PATH][0]
    written_config = config_write_call[0][1]
    assert written_config["provider"][SYNTH_PROVIDER_ID]["options"]["baseURL"] == override_url


def test_opencode_cmd_subprocess_error(runner: CliRunner, mock_env, empty_config):
    """Test that subprocess errors are handled gracefully."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-error"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json"), \
         mock.patch("synth_ai.cli.opencode.subprocess.run") as mock_run:

        # Simulate subprocess failure
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, "opencode")

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert "Failed to launch OpenCode" in result.output


def test_opencode_cmd_install_loop_success(runner: CliRunner, mock_env, empty_config):
    """Test that opencode_cmd retries installation if not found initially."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-install"

    # First call returns None, second call returns path
    find_calls = [None, mock_bin_path]

    with mock.patch("synth_ai.cli.opencode.find_bin_path", side_effect=find_calls), \
         mock.patch("synth_ai.cli.opencode.install_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json"), \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Found your installed OpenCode at {mock_bin_path}" in result.output


def test_ensure_synth_provider_in_config_empty():
    """Test _ensure_synth_provider_in_config with empty config."""
    config = {}
    url = "https://api.example.com"
    model = "synth-small"

    result = _ensure_synth_provider_in_config(config, url, model)

    assert "provider" in result
    assert SYNTH_PROVIDER_ID in result["provider"]

    synth_provider = result["provider"][SYNTH_PROVIDER_ID]
    assert synth_provider["npm"] == "@ai-sdk/openai-compatible"
    assert synth_provider["name"] == "Synth"
    assert synth_provider["options"]["baseURL"] == url
    assert model in synth_provider["models"]


def test_ensure_synth_provider_in_config_existing_provider():
    """Test _ensure_synth_provider_in_config preserves existing providers."""
    config = {
        "provider": {
            "other_provider": {
                "name": "Other",
                "models": {"model1": {}},
            }
        }
    }
    url = "https://api.example.com"
    model = "synth-small"

    result = _ensure_synth_provider_in_config(config, url, model)

    # Verify existing provider is preserved
    assert "other_provider" in result["provider"]
    assert result["provider"]["other_provider"]["name"] == "Other"

    # Verify synth provider is added
    assert SYNTH_PROVIDER_ID in result["provider"]


def test_ensure_synth_provider_in_config_updates_url():
    """Test _ensure_synth_provider_in_config updates URL when called again."""
    config = {
        "provider": {
            SYNTH_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Synth",
                "models": {"old_model": {}},
                "options": {"baseURL": "https://old-url.com"},
            }
        }
    }
    new_url = "https://new-url.com"
    new_model = "new_model"

    result = _ensure_synth_provider_in_config(config, new_url, new_model)

    # URL should be updated
    assert result["provider"][SYNTH_PROVIDER_ID]["options"]["baseURL"] == new_url
    # Old model should still exist
    assert "old_model" in result["provider"][SYNTH_PROVIDER_ID]["models"]
    # New model should be added
    assert new_model in result["provider"][SYNTH_PROVIDER_ID]["models"]


def test_ensure_synth_api_key_in_auth_file_new_file():
    """Test _ensure_synth_api_key_in_auth_file with empty auth file."""
    api_key = "test-api-key-new"
    empty_auth = {}

    with mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_auth), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write:

        _ensure_synth_api_key_in_auth_file(api_key)

    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args[0][0] == AUTH_PATH

    written_data = call_args[0][1]
    assert SYNTH_PROVIDER_ID in written_data
    assert written_data[SYNTH_PROVIDER_ID]["type"] == "api"
    assert written_data[SYNTH_PROVIDER_ID]["key"] == api_key


def test_ensure_synth_api_key_in_auth_file_already_correct():
    """Test _ensure_synth_api_key_in_auth_file when key already matches."""
    api_key = "test-api-key-same"
    existing_auth = {
        SYNTH_PROVIDER_ID: {
            "type": "api",
            "key": api_key,
        }
    }

    with mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=existing_auth), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write:

        _ensure_synth_api_key_in_auth_file(api_key)

    # Should not write if key already matches
    mock_write.assert_not_called()


def test_ensure_synth_api_key_in_auth_file_updates_key():
    """Test _ensure_synth_api_key_in_auth_file updates key when different."""
    new_api_key = "test-api-key-new"
    existing_auth = {
        SYNTH_PROVIDER_ID: {
            "type": "api",
            "key": "old-api-key",
        }
    }

    with mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=existing_auth), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write:

        _ensure_synth_api_key_in_auth_file(new_api_key)

    mock_write.assert_called_once()
    written_data = mock_write.call_args[0][1]
    assert written_data[SYNTH_PROVIDER_ID]["key"] == new_api_key


def test_ensure_synth_api_key_in_auth_file_preserves_other_providers():
    """Test _ensure_synth_api_key_in_auth_file preserves other providers."""
    api_key = "test-api-key-preserve"
    existing_auth = {
        "other_provider": {
            "type": "api",
            "key": "other-key",
        }
    }

    with mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=existing_auth), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write:

        _ensure_synth_api_key_in_auth_file(api_key)

    written_data = mock_write.call_args[0][1]
    # Other provider should be preserved
    assert "other_provider" in written_data
    assert written_data["other_provider"]["key"] == "other-key"
    # Synth provider should be added
    assert SYNTH_PROVIDER_ID in written_data


def test_opencode_cmd_sets_schema_in_config(runner: CliRunner, mock_env):
    """Test that opencode_cmd sets $schema in config."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-schema"
    config = {}

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=config), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0

    # Find config write call
    config_write_call = [call for call in mock_write.call_args_list if call[0][0] == CONFIG_PATH][0]
    written_config = config_write_call[0][1]
    assert "$schema" in written_config
    assert written_config["$schema"] == "https://opencode.ai/config.json"


def test_opencode_cmd_sets_model_in_config(runner: CliRunner, mock_env, empty_config):
    """Test that opencode_cmd sets the model in config."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-model"
    model = "synth-small"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", model])

    assert result.exit_code == 0

    # Find config write call
    config_write_call = [call for call in mock_write.call_args_list if call[0][0] == CONFIG_PATH][0]
    written_config = config_write_call[0][1]
    assert written_config["model"] == f"{SYNTH_PROVIDER_ID}/{model}"


def test_opencode_cmd_with_force_flag(runner: CliRunner, mock_env, empty_config):
    """Test that --force flag is passed to resolve_env_var."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key-force"

    with mock.patch("synth_ai.cli.opencode.find_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_opencode", return_value=True), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key) as mock_resolve, \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json"), \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small", "--force"])

    assert result.exit_code == 0

    # Verify resolve_env_var was called with force=True
    mock_resolve.assert_called_once_with("SYNTH_API_KEY", override_process_env=True)
