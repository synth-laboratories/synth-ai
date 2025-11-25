"""Unit tests for scan command components.

Tests individual components of the scan command in isolation, including
health checks, metadata extraction, data structures, and formatting.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from synth_ai.cli.commands.scan.health_checker import (
    check_app_health,
    check_multiple_apps_health,
    extract_app_info,
)
from synth_ai.cli.commands.scan.models import ScannedApp
from synth_ai.cli.commands.scan.core import format_app_table, format_app_json


class TestScannedApp:
    """Tests for ScannedApp dataclass."""

    def test_scanned_app_creation(self):
        """Test creating a ScannedApp with all fields."""
        app = ScannedApp(
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
            metadata={"key": "value"},
        )
        assert app.name == "test_app"
        assert app.url == "http://localhost:8000"
        assert app.type == "local"
        assert app.health_status == "healthy"
        assert app.port == 8000
        assert app.app_id == "test_app"
        assert app.version == "1.0.0"

    def test_scanned_app_cloudflare(self):
        """Test creating a ScannedApp for Cloudflare tunnel."""
        app = ScannedApp(
            name="test_app",
            url="https://abc123.trycloudflare.com",
            type="cloudflare",
            health_status="healthy",
            port=8000,
            tunnel_mode="quick",
            tunnel_hostname="abc123.trycloudflare.com",
            app_id="test_app",
            task_name=None,
            dataset_id=None,
            version=None,
            discovered_via="tunnel_records",
        )
        assert app.type == "cloudflare"
        assert app.tunnel_mode == "quick"
        assert app.tunnel_hostname == "abc123.trycloudflare.com"

    def test_scanned_app_minimal(self):
        """Test creating a ScannedApp with minimal fields."""
        app = ScannedApp(
            name="localhost:8000",
            url="http://localhost:8000",
            type="local",
            health_status="unknown",
            port=8000,
            tunnel_mode=None,
            tunnel_hostname=None,
            app_id=None,
            task_name=None,
            dataset_id=None,
            version=None,
            discovered_via="port_scan",
        )
        assert app.name == "localhost:8000"
        assert app.health_status == "unknown"


class TestExtractAppInfo:
    """Tests for extract_app_info function."""

    def test_extract_app_info_full(self):
        """Test extracting all fields from complete metadata."""
        metadata = {
            "service": {
                "task": {
                    "id": "banking77",
                    "name": "Banking77 Intent Classification",
                    "version": "1.0.0",
                }
            },
            "dataset": {"id": "banking77_dataset"},
        }
        app_id, task_name, dataset_id, version = extract_app_info(metadata)
        assert app_id == "banking77"
        assert task_name == "Banking77 Intent Classification"
        assert dataset_id == "banking77_dataset"
        assert version == "1.0.0"

    def test_extract_app_info_partial(self):
        """Test extracting with missing fields."""
        metadata = {
            "service": {
                "task": {
                    "id": "test_app",
                    "name": "Test App",
                }
            }
        }
        app_id, task_name, dataset_id, version = extract_app_info(metadata)
        assert app_id == "test_app"
        assert task_name == "Test App"
        assert dataset_id is None
        assert version is None

    def test_extract_app_info_empty(self):
        """Test extracting from empty metadata."""
        metadata = {}
        app_id, task_name, dataset_id, version = extract_app_info(metadata)
        assert app_id is None
        assert task_name is None
        assert dataset_id is None
        assert version is None

    def test_extract_app_info_malformed(self):
        """Test extracting from malformed metadata."""
        metadata = {"service": "not_a_dict"}
        app_id, task_name, dataset_id, version = extract_app_info(metadata)
        assert app_id is None
        assert task_name is None


class TestCheckAppHealth:
    """Tests for check_app_health function."""

    @pytest.mark.asyncio
    async def test_check_app_health_healthy(self):
        """Test health check for healthy app."""
        with mock.patch("httpx.AsyncClient") as mock_client:
            mock_response_health = mock.Mock()
            mock_response_health.status_code = 200
            mock_response_health.json.return_value = {"status": "healthy"}
            mock_response_health.headers = {"content-type": "application/json"}

            mock_response_info = mock.Mock()
            mock_response_info.status_code = 200
            mock_response_info.json.return_value = {
                "service": {"task": {"id": "test_app", "version": "1.0.0"}}
            }

            mock_client_instance = mock.AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.side_effect = [mock_response_health, mock_response_info]
            mock_client.return_value = mock_client_instance

            status, metadata = await check_app_health("http://localhost:8000", "test_key")
            assert status == "healthy"
            assert "service" in metadata

    @pytest.mark.asyncio
    async def test_check_app_health_unhealthy(self):
        """Test health check for unhealthy app."""
        with mock.patch("httpx.AsyncClient") as mock_client:
            mock_response = mock.Mock()
            mock_response.status_code = 500
            mock_response.headers = {"content-type": "application/json"}

            mock_client_instance = mock.AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            status, metadata = await check_app_health("http://localhost:8000", None)
            assert status == "unhealthy"
            assert metadata.get("http_status") == 500

    @pytest.mark.asyncio
    async def test_check_app_health_timeout(self):
        """Test health check timeout handling."""
        with mock.patch("httpx.AsyncClient") as mock_client:
            import httpx

            mock_client_instance = mock.AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value = mock_client_instance

            status, metadata = await check_app_health("http://localhost:8000", None)
            assert status == "unknown"
            assert metadata.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_check_app_health_cloudflare_error(self):
        """Test health check for Cloudflare tunnel error."""
        with mock.patch("httpx.AsyncClient") as mock_client:
            mock_response = mock.Mock()
            mock_response.status_code = 530
            mock_response.headers = {"content-type": "text/html"}

            mock_client_instance = mock.AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            status, metadata = await check_app_health("https://abc.trycloudflare.com", None)
            assert status == "unhealthy"
            assert "tunnel_error" in metadata

    @pytest.mark.asyncio
    async def test_check_app_health_healthy_flag(self):
        """Test health check with 'healthy' boolean flag."""
        with mock.patch("httpx.AsyncClient") as mock_client:
            mock_response_health = mock.Mock()
            mock_response_health.status_code = 200
            mock_response_health.json.return_value = {"healthy": True}
            mock_response_health.headers = {"content-type": "application/json"}

            mock_response_info = mock.Mock()
            mock_response_info.status_code = 200
            mock_response_info.json.return_value = {}

            mock_client_instance = mock.AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.side_effect = [mock_response_health, mock_response_info]
            mock_client.return_value = mock_client_instance

            status, _ = await check_app_health("http://localhost:8000", None)
            assert status == "healthy"


class TestCheckMultipleAppsHealth:
    """Tests for check_multiple_apps_health function."""

    @pytest.mark.asyncio
    async def test_check_multiple_apps_health(self):
        """Test concurrent health checks for multiple apps."""
        with mock.patch("synth_ai.cli.commands.scan.health_checker.check_app_health") as mock_check:
            mock_check.side_effect = [
                ("healthy", {"app": "1"}),
                ("unhealthy", {"app": "2"}),
                ("unknown", {"app": "3"}),
            ]

            urls = ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002"]
            results = await check_multiple_apps_health(urls, "test_key", timeout=1.0, max_concurrent=2)

            assert len(results) == 3
            assert results["http://localhost:8000"][0] == "healthy"
            assert results["http://localhost:8001"][0] == "unhealthy"
            assert results["http://localhost:8002"][0] == "unknown"
            assert mock_check.call_count == 3


class TestFormatAppTable:
    """Tests for format_app_table function."""

    def test_format_app_table_empty(self):
        """Test formatting empty app list."""
        result = format_app_table([])
        assert "No active task apps found" in result

    def test_format_app_table_single(self):
        """Test formatting single app."""
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
        result = format_app_table(apps)
        assert "test_app" in result
        assert "8000" in result
        assert "healthy" in result
        assert "local" in result

    def test_format_app_table_multiple(self):
        """Test formatting multiple apps."""
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
        result = format_app_table(apps)
        assert "app1" in result
        assert "app2" in result
        assert "cloudflare" in result


class TestFormatAppJson:
    """Tests for format_app_json function."""

    def test_format_app_json_empty(self):
        """Test formatting empty app list as JSON."""
        result = format_app_json([])
        data = json.loads(result)
        assert data["apps"] == []
        assert data["scan_summary"]["total_found"] == 0

    def test_format_app_json_single(self):
        """Test formatting single app as JSON."""
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
                metadata={"key": "value"},
            )
        ]
        result = format_app_json(apps)
        data = json.loads(result)
        assert len(data["apps"]) == 1
        assert data["apps"][0]["name"] == "test_app"
        assert data["apps"][0]["health_status"] == "healthy"
        assert data["scan_summary"]["total_found"] == 1
        assert data["scan_summary"]["healthy"] == 1
        assert data["scan_summary"]["local_count"] == 1

    def test_format_app_json_summary(self):
        """Test JSON summary statistics."""
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
                version=None,
                discovered_via="port_scan",
            ),
            ScannedApp(
                name="app2",
                url="http://localhost:8001",
                type="local",
                health_status="unhealthy",
                port=8001,
                tunnel_mode=None,
                tunnel_hostname=None,
                app_id="app2",
                task_name=None,
                dataset_id=None,
                version=None,
                discovered_via="port_scan",
            ),
            ScannedApp(
                name="app3",
                url="https://abc.trycloudflare.com",
                type="cloudflare",
                health_status="healthy",
                port=8002,
                tunnel_mode="quick",
                tunnel_hostname="abc.trycloudflare.com",
                app_id="app3",
                task_name=None,
                dataset_id=None,
                version=None,
                discovered_via="tunnel_records",
            ),
        ]
        result = format_app_json(apps)
        data = json.loads(result)
        assert data["scan_summary"]["total_found"] == 3
        assert data["scan_summary"]["healthy"] == 2
        assert data["scan_summary"]["unhealthy"] == 1
        assert data["scan_summary"]["local_count"] == 2
        assert data["scan_summary"]["cloudflare_count"] == 1







