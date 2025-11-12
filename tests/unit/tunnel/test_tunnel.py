"""Unit tests for Cloudflare Tunnel functionality."""
from typing import Any
from unittest.mock import Mock, patch

import pytest
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.cloudflare import (
    create_tunnel,
    open_quick_tunnel,
    stop_tunnel,
    store_tunnel_credentials,
)


class TestWhichCloudflared:
    """Tests for cloudflared binary detection.

    Note: These tests are disabled as _which_cloudflared was replaced with
    get_cloudflared_path and require_cloudflared in the refactor.
    """
    pass

    # @patch("synth_ai.cloudflare.shutil.which")
    # @patch("synth_ai.cloudflare.os.path.exists")
    # @patch("synth_ai.cloudflare.os.access")
    # def test_finds_in_path(self, mock_access, mock_exists, mock_which):
    #     """Should find cloudflared when in PATH."""
    #     mock_which.return_value = "/usr/local/bin/cloudflared"
    #     mock_exists.return_value = True
    #     mock_access.return_value = True
    #
    #     result = _which_cloudflared()
    #     assert result == "/usr/local/bin/cloudflared"
    #     mock_which.assert_called_once_with("cloudflared")
    #
    # @patch("synth_ai.cloudflare.shutil.which")
    # @patch("synth_ai.cloudflare.os.path.exists")
    # @patch("synth_ai.cloudflare.os.access")
    # def test_finds_in_common_locations(self, mock_access, mock_exists, mock_which):
    #     """Should find cloudflared in common install locations."""
    #     mock_which.return_value = None
    #     mock_exists.side_effect = lambda p: p == "/opt/homebrew/bin/cloudflared"
    #     mock_access.return_value = True
    #
    #     result = _which_cloudflared()
    #     assert result == "/opt/homebrew/bin/cloudflared"
    #
    # @patch("synth_ai.cloudflare.shutil.which")
    # @patch("synth_ai.cloudflare.os.path.exists")
    # def test_raises_when_not_found(self, mock_exists, mock_which):
    #     """Should raise FileNotFoundError when cloudflared not found."""
    #     mock_which.return_value = None
    #     mock_exists.return_value = False
    #
    #     with pytest.raises(FileNotFoundError) as exc_info:
    #         _which_cloudflared()
    #
    #     assert "cloudflared not found" in str(exc_info.value)


class TestOpenQuickTunnel:
    """Tests for opening quick tunnels."""

    @patch("synth_ai.cloudflare.require_cloudflared")
    @patch("synth_ai.cloudflare.subprocess.Popen")
    @patch("synth_ai.cloudflare.time.time")
    def test_parses_url_from_output(self, mock_time, mock_popen, mock_require):
        """Should parse trycloudflare URL from cloudflared output."""
        from pathlib import Path
        mock_require.return_value = Path("/usr/bin/cloudflared")

        # Mock process stdout
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = Mock()
        mock_proc.stdout.readline.side_effect = [
            "some output\n",
            "https://abc123.trycloudflare.com\n",
            ""
        ]
        mock_popen.return_value = mock_proc

        # Mock time for timeout
        mock_time.side_effect = [0.0, 0.1, 0.2]

        url, proc = open_quick_tunnel(8000)

        assert url == "https://abc123.trycloudflare.com"
        assert proc == mock_proc
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "/usr/bin/cloudflared"
        assert "--url" in call_args
        assert "http://127.0.0.1:8000" in call_args

    @patch("synth_ai.cloudflare.require_cloudflared")
    @patch("synth_ai.cloudflare.subprocess.Popen")
    @patch("synth_ai.cloudflare.time.time")
    def test_timeout_when_url_not_found(self, mock_time, mock_popen, mock_require):
        """Should raise RuntimeError if URL not found in timeout."""
        from pathlib import Path
        mock_require.return_value = Path("/usr/bin/cloudflared")

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = Mock()
        mock_proc.stdout.readline.return_value = "no url here\n"
        mock_proc.communicate.return_value = ("output", None)
        mock_popen.return_value = mock_proc

        # Mock time to exceed timeout
        mock_time.side_effect = [0.0, 11.0]  # Exceeds default 10s timeout

        with pytest.raises(RuntimeError) as exc_info:
            open_quick_tunnel(8000, wait_s=10.0)

        assert "Failed to parse trycloudflare URL" in str(exc_info.value)
        mock_proc.terminate.assert_called_once()


class TestStopTunnel:
    """Tests for stopping tunnel processes."""
    
    def test_stops_running_process(self):
        """Should terminate a running process."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # Still running
        
        stop_tunnel(mock_proc)
        
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
    
    def test_handles_already_stopped_process(self):
        """Should handle process that already stopped."""
        mock_proc = Mock()
        mock_proc.poll.return_value = 1  # Already exited
        
        stop_tunnel(mock_proc)
        
        mock_proc.terminate.assert_not_called()
    
    def test_handles_none(self):
        """Should handle None process gracefully."""
        stop_tunnel(None)  # Should not raise


class TestStoreTunnelCredentials:
    """Tests for storing tunnel credentials."""
    
    def test_writes_task_app_url(self, tmp_path):
        """Should write TASK_APP_URL to .env file."""
        env_file = tmp_path / ".env"
        
        store_tunnel_credentials(
            tunnel_url="https://test.trycloudflare.com",
            env_file=env_file,
        )
        
        content = env_file.read_text()
        assert "TASK_APP_URL=https://test.trycloudflare.com" in content
    
    def test_writes_access_credentials_when_provided(self, tmp_path):
        """Should write Access credentials when provided."""
        env_file = tmp_path / ".env"
        
        store_tunnel_credentials(
            tunnel_url="https://test.usesynth.ai",
            access_client_id="client-id-123",
            access_client_secret="secret-456",
            env_file=env_file,
        )
        
        content = env_file.read_text()
        assert "TASK_APP_URL=https://test.usesynth.ai" in content
        assert "CF_ACCESS_CLIENT_ID=client-id-123" in content
        assert "CF_ACCESS_CLIENT_SECRET=secret-456" in content
    
    def test_updates_existing_env_file(self, tmp_path):
        """Should update existing .env file without overwriting other vars."""
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=value\nTASK_APP_URL=old-url\n")
        
        store_tunnel_credentials(
            tunnel_url="https://new.trycloudflare.com",
            env_file=env_file,
        )
        
        content = env_file.read_text()
        assert "EXISTING_VAR=value" in content
        assert "TASK_APP_URL=https://new.trycloudflare.com" in content
        assert "old-url" not in content


class TestCloudflareTunnelDeployCfg:
    """Tests for CloudflareTunnelDeployCfg."""
    
    @patch("synth_ai.cfgs.validate_task_app")
    def test_create_with_defaults(self, mock_validate, tmp_path):
        """Should create config with default values."""
        task_app = tmp_path / "task_app.py"
        task_app.write_text("# test app\n")
        
        cfg = CloudflareTunnelDeployCfg.create(
            task_app_path=task_app,
            env_api_key="test-key",
        )
        
        assert cfg.task_app_path == task_app
        assert cfg.env_api_key == "test-key"
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8000
        assert cfg.mode == "quick"
        assert cfg.trace is True
        mock_validate.assert_called_once_with(task_app)
    
    @patch("synth_ai.cfgs.validate_task_app")
    def test_create_with_custom_values(self, mock_validate, tmp_path):
        """Should create config with custom values."""
        task_app = tmp_path / "task_app.py"
        task_app.write_text("# test app\n")
        
        cfg = CloudflareTunnelDeployCfg.create(
            task_app_path=task_app,
            env_api_key="test-key",
            host="0.0.0.0",
            port=9000,
            mode="quick",
            subdomain="my-company",
            trace=False,
        )
        
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.mode == "quick"
        assert cfg.subdomain == "my-company"
        assert cfg.trace is False
        mock_validate.assert_called_once_with(task_app)


class TestURLRegex:
    """Tests for URL regex pattern."""

    def test_matches_valid_trycloudflare_urls(self):
        """Should match valid trycloudflare URLs."""
        from synth_ai.cloudflare import _URL_RE

        valid_urls = [
            "https://abc123.trycloudflare.com",
            "https://test-xyz.trycloudflare.com",
            "https://hearts-appointments-ground-operated.trycloudflare.com",
        ]

        for url in valid_urls:
            assert _URL_RE.search(url) is not None, f"Should match {url}"

    def test_case_insensitive(self):
        """Should match URLs case-insensitively."""
        from synth_ai.cloudflare import _URL_RE

        assert _URL_RE.search("HTTPS://ABC123.TRYCLOUDFLARE.COM") is not None

    def test_does_not_match_invalid_urls(self):
        """Should not match invalid URLs."""
        from synth_ai.cloudflare import _URL_RE

        invalid_urls = [
            "http://test.trycloudflare.com",  # http not https
            "https://test.example.com",
            "https://trycloudflare.com",
            "just text",
        ]

        for url in invalid_urls:
            assert _URL_RE.search(url) is None, f"Should not match {url}"


class TestCreateTunnel:
    """Tests for create_tunnel helper."""

    @pytest.mark.asyncio
    async def test_posts_to_backend_url_base(self, monkeypatch):
        """Should call backend using the shared BACKEND_URL_BASE constant."""
        captured: dict[str, Any] = {}

        class DummyResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"tunnel_token": "token", "hostname": "cust.test"}

        class DummyClient:
            def __init__(self, *args, **kwargs):
                captured["client_kwargs"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]):
                captured["request"] = {"url": url, "headers": headers, "json": json}
                return DummyResponse()

        monkeypatch.setattr("synth_ai.cloudflare.httpx.AsyncClient", DummyClient)

        result = await create_tunnel("api-key", port=8123, subdomain="custom")
        assert result["hostname"] == "cust.test"

        from synth_ai.cloudflare import BACKEND_URL_BASE

        request = captured["request"]
        assert request["url"] == f"{BACKEND_URL_BASE}/api/v1/tunnels/"
        assert request["headers"] == {"Authorization": "Bearer api-key"}
        assert request["json"] == {
            "subdomain": "custom",
            "local_port": 8123,
            "local_host": "127.0.0.1",
        }
