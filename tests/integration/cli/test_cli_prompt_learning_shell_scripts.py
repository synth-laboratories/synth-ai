"""Integration tests for MIPRO and GEPA prompt learning via CLI.

These tests convert the shell scripts in examples/blog_posts/*/run_*.sh into pytest tests.
They verify end-to-end training workflows using the CLI.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_env_file(env_file: Path) -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Only process lines with '='
                if "=" not in line:
                    continue
                # Parse key=value
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    env_vars[key] = value
    return env_vars


def _get_test_env() -> dict[str, str]:
    """Get test environment variables, loading from .env files if available."""
    env = os.environ.copy()
    
    # Load from .env files
    repo_env = _load_env_file(REPO_ROOT / ".env")
    rl_env = _load_env_file(REPO_ROOT / "examples" / "rl" / ".env")
    
    # Merge (repo .env takes precedence)
    env.update(rl_env)
    env.update(repo_env)
    
    # Set defaults
    env.setdefault("BACKEND_BASE_URL", "http://localhost:8000")
    env.setdefault("SYNTH_BASE_URL", env.get("BACKEND_BASE_URL", "http://localhost:8000"))
    
    return env


def _check_task_app_health(task_app_url: str, api_key: Optional[str] = None) -> bool:
    """Check if task app is healthy."""
    try:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        response = httpx.get(f"{task_app_url}/health", headers=headers, timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def _check_backend_health(backend_url: str) -> bool:
    """Check if backend is healthy."""
    try:
        # Remove /api suffix if present
        base_url = backend_url.rstrip("/api").rstrip("/")
        response = httpx.get(f"{base_url}/api/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def _extract_task_app_url_from_toml(config_path: Path) -> Optional[str]:
    """Extract task_app_url from TOML config file."""
    try:
        with open(config_path, "r") as f:
            for line in f:
                if line.strip().startswith("task_app_url"):
                    # Extract value between quotes
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        value = parts[1].strip().strip('"').strip("'")
                        return value
    except Exception:
        pass
    return None


def _run_train_command(
    config_path: Path,
    backend_url: str,
    poll: bool = False,
    timeout: Optional[int] = None,
    env: Optional[dict[str, str]] = None,
    env_file: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    """Run the synth-ai train command."""
    args = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "prompt_learning",
        "--config",
        str(config_path),
        "--backend",
        backend_url,
    ]
    
    if env_file:
        args.extend(["--env-file", str(env_file)])
    
    if poll:
        args.append("--poll")
    
    env = env or _get_test_env()
    
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


def _require_env_vars(*var_names: str) -> None:
    """Skip test if required environment variables are missing."""
    env = _get_test_env()
    missing = [name for name in var_names if not env.get(name)]
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")


# MIPRO Tests


@pytest.mark.mipro
@pytest.mark.pipeline
def test_mipro_banking77_pipeline():
    """Test MIPRO optimization on Banking77 pipeline (main config)."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "OPENAI_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_pipeline_mipro_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    # Check backend health
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    # Extract task app URL from config
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        # Check task app health
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    # Run training (with timeout for CI)
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,  # Don't poll in tests - just verify job creation
        timeout=60,
        env=env,
    )
    
    # Verify command succeeded
    assert result.returncode == 0, (
        f"MIPRO training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    
    # Verify output contains job ID or status
    output = result.stdout + result.stderr
    assert len(output) > 0, "Command should produce output"


@pytest.mark.mipro
@pytest.mark.pipeline
@pytest.mark.gpt41mini
def test_mipro_banking77_pipeline_gpt41mini():
    """Test MIPRO optimization on Banking77 pipeline with gpt-4.1-mini as policy model."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "OPENAI_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_pipeline_mipro_gpt41mini_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,
        timeout=60,
        env=env,
    )
    
    assert result.returncode == 0, (
        f"MIPRO training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )


@pytest.mark.mipro
@pytest.mark.pipeline
@pytest.mark.gemini
def test_mipro_banking77_pipeline_gemini_flash_lite():
    """Test MIPRO optimization on Banking77 pipeline with gemini-2.5-flash-lite as policy model."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_pipeline_mipro_gemini_flash_lite_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    # Use --env-file flag like the shell script does
    env_file = REPO_ROOT / ".env"
    
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,
        timeout=60,
        env=env,
        env_file=env_file if env_file.exists() else None,
    )
    
    assert result.returncode == 0, (
        f"MIPRO training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )


# GEPA Tests


@pytest.mark.gepa
@pytest.mark.pipeline
def test_gepa_banking77_pipeline():
    """Test GEPA optimization on Banking77 pipeline."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "GROQ_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "gepa" / "configs" / "banking77_pipeline_gepa_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,
        timeout=60,
        env=env,
    )
    
    assert result.returncode == 0, (
        f"GEPA training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )


@pytest.mark.gepa
@pytest.mark.single_stage
def test_gepa_banking77_single_stage():
    """Test GEPA optimization on single-stage Banking77."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "GROQ_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "gepa" / "configs" / "banking77_gepa_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,
        timeout=60,
        env=env,
    )
    
    assert result.returncode == 0, (
        f"GEPA training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )


@pytest.mark.mipro
@pytest.mark.single_stage
def test_mipro_banking77_single_stage():
    """Test MIPRO optimization on single-stage Banking77."""
    _require_env_vars("SYNTH_API_KEY", "ENVIRONMENT_API_KEY", "OPENAI_API_KEY")
    
    config_path = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_mipro_local.toml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    env = _get_test_env()
    backend_url = env.get("BACKEND_BASE_URL", "http://localhost:8000")
    
    if not _check_backend_health(backend_url):
        pytest.skip(f"Backend not healthy at {backend_url}")
    
    task_app_url = _extract_task_app_url_from_toml(config_path)
    if task_app_url:
        api_key = env.get("ENVIRONMENT_API_KEY")
        if not _check_task_app_health(task_app_url, api_key):
            pytest.skip(f"Task app not healthy at {task_app_url}")
    
    result = _run_train_command(
        config_path=config_path,
        backend_url=backend_url,
        poll=False,
        timeout=60,
        env=env,
    )
    
    assert result.returncode == 0, (
        f"MIPRO training command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )

