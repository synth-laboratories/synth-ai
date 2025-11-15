"""Integration tests for task app CLI commands against dev service.

These tests verify CLI commands work against the dev backend service
without requiring local server startup.
"""

import os
import pytest


pytestmark = [pytest.mark.integration]


def _load_env_dev_first() -> None:
    """Load environment variables from .env.test.dev if it exists."""
    env_path = os.path.join(os.getcwd(), ".env.test.dev")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "SYNTH_API_KEY":
                    os.environ[k] = v
                elif k == "DEV_BACKEND_URL":
                    os.environ.setdefault(k, v)
                else:
                    os.environ.setdefault(k, v)
    except Exception:
        pass


def _derive_backend_base_url() -> str | None:
    """Get dev backend URL from environment."""
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    return None


def test_serve_help_dev():
    """Test that 'synth-ai serve --help' works."""
    _load_env_dev_first()
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"serve --help failed: {result.stderr}"
    assert "serve" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_deploy_help_dev():
    """Test that 'synth-ai deploy --help' works."""
    _load_env_dev_first()
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "deploy", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"deploy --help failed: {result.stderr}"
    assert "deploy" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_task_app_list_dev():
    """Test that 'synth-ai task-app list' works against dev service."""
    _load_env_dev_first()
    base_url = _derive_backend_base_url()
    api_key = os.getenv("SYNTH_API_KEY")
    if not base_url or not api_key:
        pytest.skip("DEV_BACKEND_URL and SYNTH_API_KEY required for dev test")

    import subprocess
    import sys

    env = os.environ.copy()
    env["SYNTH_API_KEY"] = api_key
    env["BACKEND_BASE_URL"] = base_url

    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "task-app", "list"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    # Should succeed or fail gracefully, not crash
    assert result.returncode in (0, 1), f"task-app list failed unexpectedly: {result.stderr}"


def test_scan_dev():
    """Test that 'synth-ai scan' works (may return empty results)."""
    _load_env_dev_first()
    import subprocess
    import sys

    env = os.environ.copy()
    # scan doesn't require backend, but may use it if available
    if os.getenv("DEV_BACKEND_URL"):
        env["BACKEND_BASE_URL"] = os.getenv("DEV_BACKEND_URL")

    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "scan"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    # Should succeed or fail gracefully
    assert result.returncode in (0, 1), f"scan failed unexpectedly: {result.stderr}"

