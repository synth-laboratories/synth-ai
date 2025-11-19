"""Integration tests for scan command.

Tests the scan command end-to-end, including discovery of local and tunnel apps,
health checks, and output formatting. These tests may require actual running
services or mocked HTTP responses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from synth_ai.cli.commands.scan.core import scan_command


@pytest.fixture
def runner() -> CliRunner:
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def mock_env_api_key():
    """Mock environment API key."""
    return "test_api_key_12345"


class TestScanCommandIntegration:
    """Integration tests for scan command."""

    def test_scan_command_help(self, runner: CliRunner):
        """Test scan command help output."""
        result = runner.invoke(scan_command, ["--help"])
        assert result.exit_code == 0
        assert "Scan for active Cloudflare and local task apps" in result.output
        assert "--port-range" in result.output
        assert "--json" in result.output
        assert "--verbose" in result.output

    def test_scan_command_invalid_port_range(self, runner: CliRunner):
        """Test scan command with invalid port range."""
        result = runner.invoke(scan_command, ["--port-range", "invalid"])
        assert result.exit_code != 0
        assert "Invalid port range format" in result.output

    def test_scan_command_port_range_out_of_bounds(self, runner: CliRunner):
        """Test scan command with port range out of valid bounds."""
        result = runner.invoke(scan_command, ["--port-range", "0:100"])
        assert result.exit_code != 0

        result = runner.invoke(scan_command, ["--port-range", "8000:70000"])
        assert result.exit_code != 0

        result = runner.invoke(scan_command, ["--port-range", "9000:8000"])
        assert result.exit_code != 0

    def test_scan_command_default_range(self, runner: CliRunner):
        """Test scan command with default port range."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            from synth_ai.cli.commands.scan.models import ScannedApp

            mock_scan.return_value = []
            result = runner.invoke(scan_command, [])
            assert result.exit_code == 0
            assert "No active task apps found" in result.output
            # Verify default port range was used
            call_args = mock_scan.call_args
            assert call_args[0][0] == (8000, 8100)

    def test_scan_command_custom_port_range(self, runner: CliRunner):
        """Test scan command with custom port range."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            mock_scan.return_value = []
            result = runner.invoke(scan_command, ["--port-range", "9000:9100"])
            assert result.exit_code == 0
            call_args = mock_scan.call_args
            assert call_args[0][0] == (9000, 9100)

    def test_scan_command_json_output(self, runner: CliRunner):
        """Test scan command with JSON output."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            from synth_ai.cli.commands.scan.models import ScannedApp

            apps = [
                ScannedApp(
                    name="test_app",
                    url="http://localhost:8000",
                    type="local",
                    health_status="healthy",
                    port=8000,
                    tunnel_mode=None,
                    tunnel_hostname=None,
                    app_id="test_app",
                    task_name="Test App",
                    dataset_id=None,
                    version="1.0.0",
                    discovered_via="port_scan",
                )
            ]
            mock_scan.return_value = apps

            result = runner.invoke(scan_command, ["--json"])
            assert result.exit_code == 0

            # Verify JSON output is valid
            data = json.loads(result.output)
            assert "apps" in data
            assert "scan_summary" in data
            assert len(data["apps"]) == 1
            assert data["apps"][0]["name"] == "test_app"

    def test_scan_command_verbose(self, runner: CliRunner):
        """Test scan command with verbose output."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            mock_scan.return_value = []
            result = runner.invoke(scan_command, ["--verbose"])
            assert result.exit_code == 0
            # Verbose output should show scanning progress
            # (actual output depends on implementation)

    def test_scan_command_with_api_key(self, runner: CliRunner, mock_env_api_key: str):
        """Test scan command with custom API key."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            mock_scan.return_value = []
            result = runner.invoke(scan_command, ["--api-key", mock_env_api_key])
            assert result.exit_code == 0
            # Verify API key was passed to run_scan
            call_args = mock_scan.call_args
            assert call_args[0][2] == mock_env_api_key

    def test_scan_command_table_output(self, runner: CliRunner):
        """Test scan command table output format."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            from synth_ai.cli.commands.scan.models import ScannedApp

            apps = [
                ScannedApp(
                    name="app1",
                    url="http://localhost:8000",
                    type="local",
                    health_status="healthy",
                    port=8000,
                    tunnel_mode=None,
                    tunnel_hostname=None,
                    app_id="app1",
                    task_name=None,
                    dataset_id=None,
                    version="1.0.0",
                    discovered_via="port_scan",
                ),
                ScannedApp(
                    name="app2",
                    url="https://abc.trycloudflare.com",
                    type="cloudflare",
                    health_status="unhealthy",
                    port=8001,
                    tunnel_mode="quick",
                    tunnel_hostname="abc.trycloudflare.com",
                    app_id="app2",
                    task_name=None,
                    dataset_id=None,
                    version=None,
                    discovered_via="tunnel_records",
                ),
            ]
            mock_scan.return_value = apps

            result = runner.invoke(scan_command, [])
            assert result.exit_code == 0
            assert "app1" in result.output
            assert "app2" in result.output
            assert "8000" in result.output
            assert "cloudflare" in result.output

    def test_scan_command_empty_results(self, runner: CliRunner):
        """Test scan command with no apps found."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            mock_scan.return_value = []
            result = runner.invoke(scan_command, [])
            assert result.exit_code == 0
            assert "No active task apps found" in result.output

    def test_scan_command_json_summary(self, runner: CliRunner):
        """Test scan command JSON summary statistics."""
        with mock.patch("synth_ai.cli.commands.scan.core.run_scan") as mock_scan:
            from synth_ai.cli.commands.scan.models import ScannedApp

            apps = [
                ScannedApp(
                    name="healthy_app",
                    url="http://localhost:8000",
                    type="local",
                    health_status="healthy",
                    port=8000,
                    tunnel_mode=None,
                    tunnel_hostname=None,
                    app_id="healthy_app",
                    task_name=None,
                    dataset_id=None,
                    version=None,
                    discovered_via="port_scan",
                ),
                ScannedApp(
                    name="unhealthy_app",
                    url="http://localhost:8001",
                    type="local",
                    health_status="unhealthy",
                    port=8001,
                    tunnel_mode=None,
                    tunnel_hostname=None,
                    app_id="unhealthy_app",
                    task_name=None,
                    dataset_id=None,
                    version=None,
                    discovered_via="port_scan",
                ),
                ScannedApp(
                    name="tunnel_app",
                    url="https://abc.trycloudflare.com",
                    type="cloudflare",
                    health_status="healthy",
                    port=8002,
                    tunnel_mode="quick",
                    tunnel_hostname="abc.trycloudflare.com",
                    app_id="tunnel_app",
                    task_name=None,
                    dataset_id=None,
                    version=None,
                    discovered_via="tunnel_records",
                ),
            ]
            mock_scan.return_value = apps

            result = runner.invoke(scan_command, ["--json"])
            assert result.exit_code == 0

            data = json.loads(result.output)
            summary = data["scan_summary"]
            assert summary["total_found"] == 3
            assert summary["healthy"] == 2
            assert summary["unhealthy"] == 1
            assert summary["local_count"] == 2
            assert summary["cloudflare_count"] == 1





