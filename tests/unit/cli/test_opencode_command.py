from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from synth_ai.cli.agents.opencode import (
    opencode_cmd,
    CONFIG_PATH,
    AUTH_PATH,
    SYNTH_PROVIDER_ID,
)
from synth_ai.core.urls import BACKEND_URL_SYNTH_RESEARCH_BASE


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def empty_config():
    """Return an empty config dict."""
    return {}


def test_opencode_cmd_not_found_no_install(runner: CliRunner):
    """Command exits when OpenCode missing and install declined."""
    with mock.patch("synth_ai.cli.opencode.get_bin_path", return_value=None), \
         mock.patch("synth_ai.cli.opencode.install_bin", return_value=False):

        result = runner.invoke(opencode_cmd)

    assert result.exit_code == 0
    assert "Failed to find your installed OpenCode" in result.output


def test_opencode_cmd_found_but_not_runnable(runner: CliRunner):
    """Command exits when OpenCode fails verification."""
    mock_bin_path = "/usr/local/bin/opencode"

    with mock.patch("synth_ai.cli.opencode.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_bin", return_value=False):

        result = runner.invoke(opencode_cmd)

    assert result.exit_code == 0
    assert f"Using OpenCode at {mock_bin_path}" in result.output
    assert "Failed to verify OpenCode is runnable" in result.output


def test_opencode_cmd_configures_model(runner: CliRunner, empty_config):
    """Model option writes auth and config entries."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key"

    with mock.patch("synth_ai.cli.opencode.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.opencode.write_agents_md"), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run") as mock_run:

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with([mock_bin_path], check=True)

    # Auth and config writes executed
    auth_call = next(call for call in mock_write.call_args_list if call.args[0] == AUTH_PATH)
    config_call = next(call for call in mock_write.call_args_list if call.args[0] == CONFIG_PATH)
    written_auth = auth_call.args[1]
    written_config = config_call.args[1]
    assert written_auth[SYNTH_PROVIDER_ID]["key"] == mock_api_key
    assert written_config["model"] == f"{SYNTH_PROVIDER_ID}/synth-small"
    assert written_config["provider"][SYNTH_PROVIDER_ID]["options"]["baseURL"] == BACKEND_URL_SYNTH_RESEARCH_BASE


def test_opencode_cmd_override_url(runner: CliRunner, empty_config):
    """Override URL is respected."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key"
    override_url = "https://custom.example.com/api"

    with mock.patch("synth_ai.cli.opencode.get_bin_path", return_value=mock_bin_path), \
         mock.patch("synth_ai.cli.opencode.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.opencode.write_agents_md"), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json") as mock_write, \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small", "--url", override_url])

    assert result.exit_code == 0
    config_call = next(call for call in mock_write.call_args_list if call.args[0] == CONFIG_PATH)
    written_config = config_call.args[1]
    assert written_config["provider"][SYNTH_PROVIDER_ID]["options"]["baseURL"] == override_url


def test_opencode_cmd_install_retry(runner: CliRunner, empty_config):
    """Installation is retried until successful."""
    mock_bin_path = "/usr/local/bin/opencode"
    mock_api_key = "test-api-key"

    find_calls = [None, mock_bin_path]

    with mock.patch("synth_ai.cli.opencode.get_bin_path", side_effect=find_calls), \
         mock.patch("synth_ai.cli.opencode.install_bin", return_value=True), \
         mock.patch("synth_ai.cli.opencode.verify_bin", return_value=True), \
         mock.patch("synth_ai.cli.opencode.write_agents_md"), \
         mock.patch("synth_ai.cli.opencode.resolve_env_var", return_value=mock_api_key), \
         mock.patch("synth_ai.cli.opencode.load_json_to_dict", return_value=empty_config.copy()), \
         mock.patch("synth_ai.cli.opencode.create_and_write_json"), \
         mock.patch("synth_ai.cli.opencode.subprocess.run"):

        result = runner.invoke(opencode_cmd, ["--model", "synth-small"])

    assert result.exit_code == 0
    assert f"Using OpenCode at {mock_bin_path}" in result.output
