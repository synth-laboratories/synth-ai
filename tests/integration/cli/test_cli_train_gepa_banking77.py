"""Integration test for GEPA prompt learning with Banking77 task app.

This test verifies the end-to-end flow of running GEPA optimization
on the Banking77 classification task using the test configuration.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run(args: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and return the result."""
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        cwd=Path(__file__).parent.parent.parent.parent,
    )


@pytest.mark.slow
def test_train_gepa_banking77_with_polling():
    """Test GEPA prompt learning on Banking77 with polling enabled.
    
    This test:
    1. Verifies the train command accepts the GEPA config
    2. Starts a training job with polling enabled
    3. Ensures the job can be created and tracked
    
    Note: This test requires:
    - Backend running at http://localhost:8000
    - Banking77 task app deployed (or using Modal URL)
    - GROQ_API_KEY environment variable set
    - SYNTH_API_KEY environment variable set (if using auth)
    """
    # Get config path relative to repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    config_path = repo_root / "examples" / "blog_posts" / "gepa" / "configs" / "banking77_gepa_test.toml"
    
    # Alternative: use monorepo config if blog post config doesn't exist
    if not config_path.exists():
        monorepo_config = Path(__file__).parent.parent.parent.parent.parent / "monorepo" / "backend" / "app" / "routes" / "prompt_learning" / "configs" / "banking77_gepa_test.toml"
        if monorepo_config.exists():
            config_path = monorepo_config
        else:
            pytest.skip(f"Config file not found: {config_path} or {monorepo_config}")
    
    # Set up environment
    env = os.environ.copy()
    env.setdefault("BACKEND_BASE_URL", "http://localhost:8000/api")
    
    # Run train command with polling (but with timeout to avoid hanging)
    # Note: --poll will keep running until job completes, so we use a timeout
    # In CI, you might want to use --no-poll and check job status separately
    args = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "prompt_learning",
        "--config",
        str(config_path),
        "--backend",
        "http://localhost:8000",
        "--poll",
    ]
    
    # Run with timeout (adjust based on your needs)
    # For a full test, you might want to run without --poll and check status separately
    result = _run(args, env=env, timeout=300)  # 5 minute timeout
    
    # Check that command started successfully
    # Note: If job fails or times out, we still want to verify the command works
    assert result.returncode in (0, 124), (
        f"Train command failed or timed out.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    
    # Verify we got some output (job ID, status, etc.)
    output = result.stdout + result.stderr
    assert len(output) > 0, "Command should produce some output"


@pytest.mark.slow
def test_train_gepa_banking77_no_poll():
    """Test GEPA prompt learning on Banking77 without polling.
    
    This variant creates the job and returns immediately,
    which is better for CI environments.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    config_path = repo_root / "examples" / "blog_posts" / "gepa" / "configs" / "banking77_gepa_test.toml"
    
    if not config_path.exists():
        monorepo_config = Path(__file__).parent.parent.parent.parent.parent / "monorepo" / "backend" / "app" / "routes" / "prompt_learning" / "configs" / "banking77_gepa_test.toml"
        if monorepo_config.exists():
            config_path = monorepo_config
        else:
            pytest.skip(f"Config file not found: {config_path} or {monorepo_config}")
    
    env = os.environ.copy()
    env.setdefault("BACKEND_BASE_URL", "http://localhost:8000/api")
    
    args = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "prompt_learning",
        "--config",
        str(config_path),
        "--backend",
        "http://localhost:8000",
        # No --poll flag
    ]
    
    result = _run(args, env=env, timeout=60)  # 1 minute timeout
    
    # Should succeed and return job ID
    assert result.returncode == 0, (
        f"Train command failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    
    # Should output job ID or status
    output = result.stdout + result.stderr
    assert len(output) > 0, "Command should produce output with job ID"

